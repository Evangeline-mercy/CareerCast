"""
CareerCast - Milestone 2 Flask API

Pipeline:
    Resume PDF/DOCX
        ?
    Resume Extraction
        ?
    SBERT (384-dimensional embedding)
        ?
    Logistic Regression / Random Forest / XGBoost
        ?
    Top-K Career Predictions
        ?
    JSON API Response
"""

from pathlib import Path
import tempfile
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

from resume_extractor import analyze_resume
from live_milestone2_predictor import predict_resume


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Allow frontend requests
CORS(app)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename):
    """Check whether the uploaded file is PDF or DOCX."""

    if not filename:
        return False

    extension = Path(filename).suffix.lower()

    return extension in ALLOWED_EXTENSIONS


def create_error_response(message, status_code=400):
    """Return a consistent API error response."""

    return jsonify({
        "success": False,
        "error": message,
    }), status_code


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "success": True,
        "service": "CareerCast Milestone 2 API",
        "status": "running",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "career_classes": 96,
        "models": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
        ],
    })


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "status": "healthy",
        "service": "CareerCast Milestone 2",
    })


# ============================================================
# RESUME PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    temp_file = None

    try:

        # ----------------------------------------------------
        # Check uploaded file
        # ----------------------------------------------------

        if "resume" not in request.files:

            return create_error_response(
                "No resume file uploaded. "
                "Use form-data field name 'resume'.",
                400,
            )

        uploaded_file = request.files["resume"]

        if not uploaded_file.filename:

            return create_error_response(
                "Uploaded file has no filename.",
                400,
            )

        # ----------------------------------------------------
        # Validate extension
        # ----------------------------------------------------

        if not allowed_file(
            uploaded_file.filename
        ):

            return create_error_response(
                "Unsupported resume format. "
                "Only PDF and DOCX are supported.",
                400,
            )

        # ----------------------------------------------------
        # Save temporary file
        # ----------------------------------------------------

        extension = Path(
            uploaded_file.filename
        ).suffix.lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp:

            uploaded_file.save(
                temp.name
            )

            temp_file = Path(
                temp.name
            )

        # ----------------------------------------------------
        # Extract resume
        # ----------------------------------------------------

        resume_data = analyze_resume(
            temp_file
        )

        resume_text = resume_data["text"]

        skills = resume_data["skills"]

        # ----------------------------------------------------
        # Career prediction
        # ----------------------------------------------------

        prediction_result = predict_resume(
            resume_text,
            top_k=5,
        )

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "resume": {
                "file_name": resume_data[
                    "file_name"
                ],

                "file_type": resume_data[
                    "file_type"
                ],

                "text_length": resume_data[
                    "text_length"
                ],

                "skill_count": resume_data[
                    "skill_count"
                ],

                "skills": skills,
            },

            "prediction": {

                "embedding_dimension":
                    prediction_result[
                        "embedding_dimension"
                    ],

                "models":
                    prediction_result[
                        "models"
                    ],
            },

        }), 200

    except FileNotFoundError as error:

        return create_error_response(
            str(error),
            404,
        )

    except ValueError as error:

        return create_error_response(
            str(error),
            400,
        )

    except Exception as error:

        print("\n" + "=" * 70)
        print("UNEXPECTED API ERROR")
        print("=" * 70)

        traceback.print_exc()

        return create_error_response(
            f"Internal server error: {str(error)}",
            500,
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded file
        # ----------------------------------------------------

        if temp_file is not None:

            try:

                if temp_file.exists():

                    temp_file.unlink()

            except Exception:

                pass


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CAREERCAST - MILESTONE 2 FLASK API")
    print("=" * 70)

    print("\nAPI endpoints:")
    print("  GET  /")
    print("  GET  /health")
    print("  POST /predict")

    print("\nSupported files:")
    print("  PDF")
    print("  DOCX")

    print("\nML pipeline:")
    print("  all-MiniLM-L6-v2")
    print("  384-dimensional embedding")
    print("  96 career classes")

    print("\nStarting Flask server...")
    print("URL: http://127.0.0.1:5000")

    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )