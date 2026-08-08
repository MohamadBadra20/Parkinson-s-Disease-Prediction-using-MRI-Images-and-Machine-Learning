import io
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
from flask import Flask, render_template, request, redirect, session, url_for, send_file, send_from_directory, abort
from nilearn.image import resample_to_img
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "database.db"
MODEL_DIR = BASE_DIR / "soft_voting_deployment"
UPLOAD_FOLDER = BASE_DIR / "uploads"
PREVIEW_FOLDER = BASE_DIR / "previews"
IMAGE_FOLDER = BASE_DIR / "Images"

UPLOAD_FOLDER.mkdir(exist_ok=True)
PREVIEW_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("NEUROVISION_SECRET_KEY", "change-this-secret-key-before-production")
app.config.update(
    MAX_CONTENT_LENGTH=500 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("NEUROVISION_HTTPS", "0") == "1",
)

SESSION_TIMEOUT = timedelta(minutes=10)
ABSOLUTE_SESSION_LIMIT = timedelta(hours=10)
SV_THRESHOLD = float(os.environ.get("NEUROVISION_THRESHOLD", "0.35"))
ADMIN_USERNAME = os.environ.get("NEUROVISION_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("NEUROVISION_ADMIN_PASSWORD", "NeuroAdmin##")

ALLOWED_EXTENSIONS = (".nii", ".nii.gz")


def db():
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, age INTEGER NOT NULL,
        prediction TEXT NOT NULL, probability REAL NOT NULL, created_at TEXT NOT NULL)""")
    # Upgrade older databases without losing existing records.
    patient_columns = {row[1] for row in cur.execute("PRAGMA table_info(patients)").fetchall()}
    if "created_by" not in patient_columns:
        cur.execute("ALTER TABLE patients ADD COLUMN created_by TEXT")
    cur.execute("""CREATE TABLE IF NOT EXISTS logins (
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, role TEXT NOT NULL, login_time TEXT NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS contact_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT NOT NULL,
        message TEXT NOT NULL, created_at TEXT NOT NULL)""")
    conn.commit()
    conn.close()


init_db()


def load_models():
    config_path = MODEL_DIR / "config.pkl"
    if not config_path.exists():
        return None, None, f"Model package not found: {MODEL_DIR}"
    try:
        config = joblib.load(config_path)
        loaded = {
            name: joblib.load(MODEL_DIR / f"{name}.pkl")
            for name in config["models"]
        }
        return config, loaded, None
    except Exception as exc:
        return None, None, f"Could not load model package: {exc}"


config, models, MODEL_ERROR = load_models()

# The deployment package is authoritative for the trained ensemble threshold.
# An environment variable can still override it for controlled experiments.
if config and "threshold" in config and "NEUROVISION_THRESHOLD" not in os.environ:
    SV_THRESHOLD = float(config["threshold"])


def format_probability(prob):
    try:
        value = float(prob)
        if value <= 1:
            value *= 100
        return round(value, 2)
    except (TypeError, ValueError):
        return prob


def is_session_expired():
    if "username" not in session:
        return True
    try:
        last_dt = datetime.strptime(session["last_activity"], "%Y-%m-%d %H:%M:%S")
        login_dt = datetime.strptime(session["login_time"], "%Y-%m-%d %H:%M:%S")
    except (KeyError, ValueError, TypeError):
        return True
    now = datetime.now()
    return (now - last_dt > SESSION_TIMEOUT) or (now - login_dt > ABSOLUTE_SESSION_LIMIT)


@app.before_request
def session_management():
    if request.endpoint in {"static", "images", "preview_file"}:
        return
    if "username" not in session:
        return
    if is_session_expired():
        session.clear()
        return redirect(url_for("login"))
    session["last_activity"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def login_session(username, role):
    now = datetime.now()
    session.clear()
    session.permanent = True
    session["username"] = username
    session["role"] = role
    session["login_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
    session["last_activity"] = session["login_time"]
    conn = db()
    conn.execute(
        "INSERT INTO logins (username, role, login_time) VALUES (?, ?, ?)",
        (username, role, now.strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    conn.close()


def allowed_file(filename):
    lower = filename.lower()
    return lower.endswith(ALLOWED_EXTENSIONS)


def validate_name(name):
    return bool(name) and len(name) <= 100 and not any(ch.isdigit() for ch in name)


def preprocess_fmri(fmri_path):
    if MODEL_ERROR:
        raise RuntimeError(MODEL_ERROR)

    fmri_img = nib.load(str(fmri_path))
    atlas_img = nib.load(str(MODEL_DIR / "AAL.nii"))

    if len(fmri_img.shape) != 4:
        raise ValueError(f"Expected a 4D fMRI file, but the uploaded image has shape {fmri_img.shape}.")
    if fmri_img.shape[3] < 2:
        raise ValueError("The fMRI file must contain multiple time points.")

    atlas_resampled = resample_to_img(
        source_img=atlas_img,
        target_img=fmri_img,
        interpolation="nearest",
    )
    fmri_data = fmri_img.get_fdata(dtype=np.float32)
    atlas_data = atlas_resampled.get_fdata(dtype=np.float32)

    region_ids = np.unique(atlas_data)
    region_ids = region_ids[region_ids != 0]
    region_ids = np.sort(region_ids)

    if len(region_ids) != 116:
        raise ValueError(
            f"Expected exactly 116 AAL regions after resampling, but found {len(region_ids)}. "
            "Check that the submitted fMRI is in the same space expected by the trained model."
        )

    roi_time_series = []
    for region in region_ids:
        mask = atlas_data == region
        if not np.any(mask):
            raise ValueError(f"AAL region {region:g} contains no voxels in the submitted image.")
        roi_time_series.append(np.nanmean(fmri_data[mask], axis=0))

    roi_time_series = np.asarray(roi_time_series, dtype=np.float64).T
    if not np.isfinite(roi_time_series).all():
        raise ValueError("The fMRI contains invalid values in the ROI time series.")

    conn_matrix = np.corrcoef(roi_time_series, rowvar=False)
    if not np.isfinite(conn_matrix).all():
        raise ValueError("Functional connectivity produced invalid values. Check the scan quality and preprocessing.")

    features = conn_matrix[np.triu_indices(116, k=1)]
    if features.shape != (6670,):
        raise ValueError(f"Expected 6,670 connectivity features, got {features.size}.")
    return features


def soft_vote_predict(features):
    if not models or not config:
        raise RuntimeError(MODEL_ERROR or "The model package is unavailable.")

    X_input = features.reshape(1, -1)
    probs = []
    for name in config["models"]:
        model = models[name]
        if not hasattr(model, "predict_proba"):
            raise RuntimeError(f"Model '{name}' does not expose predict_proba().")
        probs.append(float(model.predict_proba(X_input)[:, 1][0]))

    avg_prob = float(np.mean(probs))
    prediction = "Positive" if avg_prob >= SV_THRESHOLD else "Negative"
    return prediction, round(avg_prob * 100, 2)


def create_preview(fmri_path, patient_id):
    """Create a simple central axial slice for visual reference, not diagnosis."""
    try:
        from PIL import Image, ImageOps

        img = nib.as_closest_canonical(nib.load(str(fmri_path)))
        data = img.get_fdata(dtype=np.float32)
        volume = data[..., data.shape[3] // 2]
        z = volume.shape[2] // 2
        slice_2d = np.asarray(volume[:, :, z], dtype=np.float32)
        slice_2d = np.nan_to_num(slice_2d)
        lo, hi = np.percentile(slice_2d, [2, 98])
        if hi <= lo:
            return None
        normalized = np.clip((slice_2d - lo) / (hi - lo), 0, 1)
        arr = (normalized * 255).astype(np.uint8)
        image = Image.fromarray(arr).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        path = PREVIEW_FOLDER / f"patient_{patient_id}.png"
        image.save(path, optimize=True)
        return path.name
    except Exception:
        return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/Images/<path:filename>")
def images(filename):
    return send_from_directory(str(IMAGE_FOLDER), filename)


@app.route("/previews/<path:filename>")
def preview_file(filename):
    safe = secure_filename(filename)
    path = PREVIEW_FOLDER / safe
    if not path.is_file():
        abort(404)
    return send_from_directory(str(PREVIEW_FOLDER), safe)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    success = False
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()
        if not validate_name(name):
            error = "Please enter a valid name."
        elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            error = "Please enter a valid email address."
        elif not message or len(message) > 3000:
            error = "Please enter a message of up to 3,000 characters."
        else:
            conn = db()
            conn.execute(
                "INSERT INTO contact_messages (name,email,message,created_at) VALUES (?,?,?,?)",
                (name, email, message, datetime.now().strftime("%Y-%m-%d %H:%M")),
            )
            conn.commit()
            conn.close()
            success = True
    return render_template("contact.html", success=success, error=error)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    message = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,50}", username):
            message = "Username must be 3–50 characters and use only letters, numbers, _, ., or -."
        elif password != confirm:
            message = "Passwords do not match."
        elif len(password) < 8:
            message = "Password must be at least 8 characters."
        elif not re.search(r"[A-Z]", password):
            message = "Password needs at least one uppercase letter."
        elif not re.search(r"[a-z]", password):
            message = "Password needs at least one lowercase letter."
        elif not re.search(r"[0-9]", password):
            message = "Password needs at least one number."
        elif not re.search(r"[!@#$%^&*()_+\-=?.]", password):
            message = "Password needs at least one special character."
        else:
            conn = db()
            exists = conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
            if exists:
                message = "Username already exists. Please sign in instead."
                conn.close()
            else:
                conn.execute(
                    "INSERT INTO users (username,password) VALUES (?,?)",
                    (username, generate_password_hash(password)),
                )
                conn.commit()
                conn.close()
                login_session(username, "user")
                return redirect(url_for("home"))
    return render_template("signup.html", message=message)


@app.route("/login", methods=["GET", "POST"])
def login():
    message = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            login_session(username, "admin")
            return redirect(url_for("dashboard"))

        conn = db()
        user = conn.execute("SELECT id,password FROM users WHERE username=?", (username,)).fetchone()
        conn.close()
        if user:
            stored = user["password"]
            valid = False
            try:
                valid = check_password_hash(stored, password)
            except (ValueError, TypeError):
                valid = stored == password  # backward compatibility with the old database
            if valid:
                # Upgrade old plaintext password on successful login.
                if not str(stored).startswith(("scrypt:", "pbkdf2:")):
                    conn = db()
                    conn.execute("UPDATE users SET password=? WHERE id=?", (generate_password_hash(password), user["id"]))
                    conn.commit()
                    conn.close()
                login_session(username, "user")
                return redirect(url_for("home"))
        message = "Invalid username or password."
    return render_template("login.html", message=message)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/upload")
def upload():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("upload.html")


@app.route("/submit", methods=["POST"])
def submit():
    if "username" not in session:
        return redirect(url_for("login"))

    name = request.form.get("name", "").strip()
    age_raw = request.form.get("age", "").strip()
    file = request.files.get("image")
    errors = []

    if not validate_name(name):
        errors.append("Enter a valid patient name without numbers.")
    try:
        age = int(age_raw)
        if not 1 <= age <= 120:
            raise ValueError
    except ValueError:
        age = age_raw
        errors.append("Age must be a number between 1 and 120.")

    if not file or not file.filename:
        errors.append("Please upload an fMRI file (.nii or .nii.gz).")
    elif not allowed_file(file.filename):
        errors.append("Only .nii and .nii.gz files are accepted.")

    if MODEL_ERROR:
        errors.append("The trained model package is not available on this server.")
    elif not (MODEL_DIR / "AAL.nii").exists():
        errors.append("The AAL atlas file is missing from the model deployment folder.")

    if errors:
        return render_template("upload.html", errors=errors, name=name, age=age)

    filename = secure_filename(file.filename)
    # Add a unique prefix so simultaneous uploads cannot overwrite each other.
    temp_path = UPLOAD_FOLDER / f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{filename}"
    file.save(temp_path)

    try:
        features = preprocess_fmri(temp_path)
        prediction, probability = soft_vote_predict(features)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        return render_template(
            "upload.html",
            errors=[f"Analysis could not be completed: {exc}"],
            name=name,
            age=age,
        )

    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO patients(name,age,prediction,probability,created_at,created_by) VALUES(?,?,?,?,?,?)",
        (name, age, prediction, probability, datetime.now().strftime("%Y-%m-%d %H:%M"), session.get("username")),
    )
    patient_id = cur.lastrowid
    conn.commit()
    conn.close()

    create_preview(temp_path, patient_id)
    if temp_path.exists():
        temp_path.unlink()

    return redirect(url_for("result", patient_id=patient_id))


@app.route("/result/<int:patient_id>")
def result(patient_id):
    if "username" not in session:
        return redirect(url_for("login"))
    conn = db()
    row = conn.execute(
        "SELECT name,age,prediction,probability,created_at FROM patients WHERE id=? AND (created_by=? OR ?='admin')",
        (patient_id, session.get("username"), session.get("role")),
    ).fetchone()
    conn.close()
    if not row:
        return redirect(url_for("upload"))

    preview_name = f"patient_{patient_id}.png"
    preview_path = preview_name if (PREVIEW_FOLDER / preview_name).exists() else None
    return render_template(
        "result.html",
        result={
            "id": patient_id,
            "name": row["name"],
            "age": row["age"],
            "prediction": row["prediction"],
            "probability": format_probability(row["probability"]),
            "created_at": row["created_at"],
            "preview_path": preview_path,
        },
    )


@app.route("/download_pdf/<int:patient_id>")
def download_pdf(patient_id):
    if "username" not in session:
        return redirect(url_for("login"))
    conn = db()
    row = conn.execute(
        "SELECT name,age,prediction,probability,created_at FROM patients WHERE id=? AND (created_by=? OR ?='admin')",
        (patient_id, session.get("username"), session.get("role")),
    ).fetchone()
    conn.close()
    if not row:
        return redirect(url_for("upload"))

    probability = format_probability(row["probability"])
    level = "Lower model probability" if probability < 30 else "Intermediate model probability" if probability < 80 else "Higher model probability"

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    pdf.setFillColorRGB(0.06, 0.19, 0.37)
    pdf.rect(0, height - 95, width, 95, fill=True, stroke=False)
    pdf.setFillColorRGB(1, 1, 1)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(45, height - 45, "NeuroVision Analysis Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(45, height - 65, "AI-assisted Parkinson's disease research prototype")

    y = height - 130
    pdf.setFillColorRGB(0.06, 0.41, 0.88)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(45, y, "Patient information")
    y -= 24
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica", 11)
    for label, value in [("Patient name", row["name"]), ("Age", row["age"]), ("Report date", row["created_at"])]:
        pdf.drawString(45, y, f"{label}: {value}")
        y -= 18

    y -= 18
    pdf.setFillColorRGB(0.06, 0.41, 0.88)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(45, y, "Model result")
    y -= 24
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica", 11)
    pdf.drawString(45, y, f"Classification: {row['prediction']}")
    y -= 18
    pdf.drawString(45, y, f"Positive-class probability: {probability}%")
    y -= 18
    pdf.drawString(45, y, f"Probability band: {level}")

    y -= 38
    pdf.setFillColorRGB(0.06, 0.41, 0.88)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(45, y, "Important interpretation note")
    y -= 24
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica", 10.5)
    note = (
        "This report records the output of the configured machine-learning pipeline. "
        "It is not a medical diagnosis. The result should be reviewed by a qualified "
        "healthcare professional together with clinical history, neurological examination "
        "and other appropriate investigations. Do not use this report alone to make "
        "treatment or other medical decisions."
    )
    words, line = note.split(), ""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    for word in words:
        test = (line + " " + word).strip()
        if stringWidth(test, "Helvetica", 10.5) <= 500:
            line = test
        else:
            pdf.drawString(45, y, line)
            y -= 16
            line = word
    if line:
        pdf.drawString(45, y, line)
        y -= 16

    y -= 20
    pdf.setFillColorRGB(0.06, 0.19, 0.37)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(45, y, "NeuroVision")
    y -= 17
    pdf.setFillColorRGB(0.35, 0.38, 0.43)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(45, y, "Research-use prototype · Please keep this report confidential.")
    pdf.save()
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{secure_filename(row['name'])}_NeuroVision_Report.pdf",
        mimetype="application/pdf",
    )


@app.route("/dashboard")
def dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("home"))
    conn = db()
    page_patients = max(request.args.get("page_patients", 1, type=int), 1)
    page_logins = max(request.args.get("page_logins", 1, type=int), 1)
    per_page = 10

    patient_search = request.args.get("patient_search", "").strip()
    patient_date = request.args.get("patient_date", "").strip()
    pq, pp = "SELECT * FROM patients WHERE 1=1", []
    if patient_search:
        pq += " AND name LIKE ?"; pp.append(f"%{patient_search}%")
    if patient_date:
        pq += " AND created_at LIKE ?"; pp.append(f"{patient_date}%")
    total_patients = conn.execute(pq.replace("SELECT *", "SELECT COUNT(*)"), pp).fetchone()[0]
    pq += " ORDER BY id DESC LIMIT ? OFFSET ?"; pp += [per_page, (page_patients - 1) * per_page]
    patients = conn.execute(pq, pp).fetchall()

    search = request.args.get("search", "").strip()
    date = request.args.get("date", "").strip()
    lq, lp = "SELECT * FROM logins WHERE 1=1", []
    if search:
        lq += " AND username LIKE ?"; lp.append(f"%{search}%")
    if date:
        lq += " AND login_time LIKE ?"; lp.append(f"{date}%")
    total_logins = conn.execute(lq.replace("SELECT *", "SELECT COUNT(*)"), lp).fetchone()[0]
    lq += " ORDER BY id DESC LIMIT ? OFFSET ?"; lp += [per_page, (page_logins - 1) * per_page]
    logins = conn.execute(lq, lp).fetchall()
    conn.close()

    return render_template(
        "dashboard.html",
        patients=patients,
        logins=logins,
        search=search,
        date=date,
        patient_search=patient_search,
        patient_date=patient_date,
        page_patients=page_patients,
        total_patient_pages=max((total_patients + per_page - 1) // per_page, 1),
        page_logins=page_logins,
        total_login_pages=max((total_logins + per_page - 1) // per_page, 1),
        total_patients=total_patients,
        total_logins=total_logins,
    )


@app.route("/messages")
def messages():
    if session.get("role") != "admin":
        return redirect(url_for("home"))
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = 5
    conn = db()
    total = conn.execute("SELECT COUNT(*) FROM contact_messages").fetchone()[0]
    rows = conn.execute(
        "SELECT id,name,email,message,created_at FROM contact_messages ORDER BY id DESC LIMIT ? OFFSET ?",
        (per_page, (page - 1) * per_page),
    ).fetchall()
    conn.close()
    return render_template("messages.html", messages=rows, page=page, total_pages=max((total + per_page - 1) // per_page, 1))


@app.route("/delete_message/<int:id>")
def delete_message(id):
    if session.get("role") != "admin":
        return redirect(url_for("home"))
    conn = db(); conn.execute("DELETE FROM contact_messages WHERE id=?", (id,)); conn.commit(); conn.close()
    return redirect(url_for("messages"))


@app.route("/clear_logins", methods=["POST"])
def clear_logins():
    if session.get("role") != "admin":
        return redirect(url_for("home"))
    conn = db(); conn.execute("DELETE FROM logins"); conn.commit(); conn.close()
    return redirect(url_for("dashboard"))


@app.route("/clear_patients", methods=["POST"])
def clear_patients():
    if session.get("role") != "admin":
        return redirect(url_for("home"))
    conn = db(); conn.execute("DELETE FROM patients"); conn.commit(); conn.close()
    for p in PREVIEW_FOLDER.glob("patient_*.png"):
        try: p.unlink()
        except OSError: pass
    return redirect(url_for("dashboard"))


@app.route("/edit_patient/<int:id>", methods=["GET", "POST"])
def edit_patient(id):
    if session.get("role") != "admin":
        return redirect(url_for("home"))
    conn = db()
    row = conn.execute("SELECT * FROM patients WHERE id=?", (id,)).fetchone()
    if not row:
        conn.close()
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        try:
            age = int(request.form.get("age", ""))
            probability = float(request.form.get("probability", ""))
        except ValueError:
            return render_template("edit_patient.html", patient=row)
        prediction = request.form.get("prediction", "")
        if not validate_name(name) or not 1 <= age <= 120 or prediction not in {"Positive", "Negative"} or not 0 <= probability <= 100:
            conn.close()
            return render_template("edit_patient.html", patient=row)
        conn.execute(
            "UPDATE patients SET name=?,age=?,prediction=?,probability=? WHERE id=?",
            (name, age, prediction, probability, id),
        )
        conn.commit()
        conn.close()
        return redirect(url_for("dashboard"))
    conn.close()
    return render_template("edit_patient.html", patient=row)


@app.route("/delete_patient/<int:id>")
def delete_patient(id):
    if session.get("role") != "admin":
        return redirect(url_for("home"))
    conn = db(); conn.execute("DELETE FROM patients WHERE id=?", (id,)); conn.commit(); conn.close()
    preview = PREVIEW_FOLDER / f"patient_{id}.png"
    if preview.exists(): preview.unlink()
    return redirect(url_for("dashboard"))


@app.route("/delete_login/<int:id>")
def delete_login(id):
    if session.get("role") != "admin":
        return redirect(url_for("home"))
    conn = db(); conn.execute("DELETE FROM logins WHERE id=?", (id,)); conn.commit(); conn.close()
    return redirect(url_for("dashboard"))


@app.errorhandler(413)
def too_large(_):
    return render_template("upload.html", errors=["The uploaded file is larger than the 500 MB limit."]), 413


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
