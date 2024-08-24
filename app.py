from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_from_directory,
)
import os
from werkzeug.utils import secure_filename
import processing

app = Flask(__name__)
app.secret_key = "test"

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result = None  # Initialize result variable
    processed = False  # Initialize processed variable
    if request.method == "POST":
        file = request.files["file"]
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            result = processing.process_video(file_path)
            session["result"] = result
            session["processed"] = True
            session["uploaded_file"] = filename  # Store filename in session

            if "number_plates" not in session:
                session["number_plates"] = []
            session["number_plates"].append(
                {"filename": filename, "number_plate": result}
            )

            return redirect(
                url_for("index")
            )  # Redirect to prevent form resubmission issues
    result = session.pop("result", None)  # Retrieve and clear the result from session
    processed = session.pop(
        "processed", False
    )  # Retrieve and clear the processed flag from session
    return render_template("index.html", result=result, processed=processed)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
