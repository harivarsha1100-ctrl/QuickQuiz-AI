"""
Smart AI MCQ Generator - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
from dotenv import load_dotenv
from typing import List, Optional
import uuid
from datetime import datetime
from modules.distractor_generator import DistractorGenerator
from modules.file_processor import FileProcessor
from modules.question_generator import QuestionGenerator
from modules.quiz_evaluator import QuizEvaluator
from modules.exporter import ResultExporter
from modules.database import SupabaseClient

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Smart AI MCQ Generator API",
    description="AI-powered system for automatic question generation and evaluation",
    version="1.0.0"
)


# CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
file_processor = FileProcessor()
question_generator = QuestionGenerator()
distractor_generator = DistractorGenerator()
quiz_evaluator = QuizEvaluator()
exporter = ResultExporter()
db_client = SupabaseClient()

# Create temp directory for file processing
os.makedirs("temp", exist_ok=True)
os.makedirs("exports", exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Smart AI MCQ Generator",
        "version": "1.0.0"
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Upload and process educational material
    Accepts: PDF, DOCX, TXT files
    Returns: File ID and extracted text preview
    """
    try:
        # Validate file type
        allowed_types = [".pdf", ".docx", ".txt"]
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_types)}"
            )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Save temporarily
        file_id = str(uuid.uuid4())
        temp_path = f"temp/{file_id}{file_ext}"

        with open(temp_path, "wb") as f:
            f.write(content)

        # Extract text
        extracted_text = file_processor.extract_text(temp_path, file_ext)

        if not extracted_text or len(extracted_text.strip()) < 100:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail="Insufficient text content. Please upload a file with more content."
            )

        # Store in database
        file_record = await db_client.create_file_record(
            file_id=file_id,
            user_id=user_id,
            file_name=file.filename,
            file_type=file_ext[1:],
            file_size=file_size,
            storage_path=temp_path,
            extracted_text=extracted_text
        )

        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "file_name": file.filename,
            "text_length": len(extracted_text),
            "preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/generate-questions")
async def generate_questions(
    file_id: str = Form(...),
    num_questions: int = Form(10),
    difficulty: Optional[str] = Form(None)
):
    """
    Generate MCQ questions from uploaded file
    Uses T5 for question generation and RoBERTa for answer extraction
    """
    try:
        # Get file from database
        file_data = await db_client.get_file(file_id)

        if not file_data:
            raise HTTPException(status_code=404, detail="File not found")

        extracted_text = file_data.get("extracted_text")

        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text content available")

        # Update processing status
        await db_client.update_file_status(file_id, "processing")

        # Generate questions
        questions = await question_generator.generate_mcqs(
            text=extracted_text,
            num_questions=num_questions,
            difficulty=difficulty
        )
        print(f"DEBUG: Generated {len(questions)} candidate questions for file {file_id}")

        # Store questions in database
        stored_questions = []
        for q in questions:
            question_id = await db_client.create_question(
                file_id=file_id,
                question_data=q
            )
            q['id'] = question_id
            stored_questions.append(q)

        # Update processing status
        await db_client.update_file_status(file_id, "completed")

        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "questions_generated": len(stored_questions),
            "questions": stored_questions
        })

    except HTTPException:
        raise
    except Exception as e:
        await db_client.update_file_status(file_id, "failed")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


@app.post("/api/create-quiz-session")
async def create_quiz_session(
    file_id: str = Form(...),
    user_id: Optional[str] = Form(None),
    session_name: Optional[str] = Form(None)
):
    """
    Create a new quiz session
    """
    try:
        # Get questions for this file
        questions = await db_client.get_questions_by_file(file_id)

        if not questions:
            raise HTTPException(status_code=404, detail="No questions found for this file")

        # Create session
        session_id = await db_client.create_quiz_session(
            file_id=file_id,
            user_id=user_id,
            session_name=session_name or f"Quiz - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            total_questions=len(questions)
        )

        # Return questions without correct answers
        quiz_questions = []
        for q in questions:
            quiz_questions.append({
                "id": q["id"],
                "question_text": q["question_text"],
                "option_a": q["option_a"],
                "option_b": q["option_b"],
                "option_c": q["option_c"],
                "option_d": q["option_d"],
                "difficulty_level": q.get("difficulty_level"),
                "blooms_taxonomy": q.get("blooms_taxonomy")
            })

        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "total_questions": len(quiz_questions),
            "questions": quiz_questions
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


@app.post("/api/submit-quiz")
async def submit_quiz(
    session_id: str = Form(...),
    answers: str = Form(...)  # JSON string of {question_id: answer}
):
    """
    Submit quiz answers and get evaluation
    """
    try:
        import json

        # Parse answers
        user_answers = json.loads(answers)

        # Get session details
        session = await db_client.get_quiz_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Quiz session not found")

        # Get all questions for this session
        questions = await db_client.get_questions_by_file(session["file_id"])

        # Evaluate answers
        results = []
        correct_count = 0

        for question in questions:
            q_id = question["id"]
            user_answer = user_answers.get(q_id)
            correct_answer = question["correct_answer"]

            is_correct = user_answer == correct_answer
            if is_correct:
                correct_count += 1

            # Store response
            await db_client.create_quiz_response(
                session_id=session_id,
                question_id=q_id,
                user_answer=user_answer,
                is_correct=is_correct
            )

            results.append({
                "question_id": q_id,
                "question_text": question["question_text"],
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "explanation": question.get("explanation", ""),
                "options": {
                    "A": question["option_a"],
                    "B": question["option_b"],
                    "C": question["option_c"],
                    "D": question["option_d"]
                }
            })

        # Update session status
        await db_client.complete_quiz_session(session_id)

        total = len(questions)
        percentage = (correct_count / total * 100) if total > 0 else 0

        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "total_questions": total,
            "correct_answers": correct_count,
            "incorrect_answers": total - correct_count,
            "percentage": round(percentage, 2),
            "results": results
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz submission failed: {str(e)}")


@app.post("/api/export")
async def export_results(
    session_id: str = Form(...),
    export_type: str = Form(...),  # "questions_only" or "results_with_answers"
    file_format: str = Form("pdf")  # "pdf" or "docx"
):
    """
    Export quiz questions or results
    """
    try:
        # Get session and questions
        session = await db_client.get_quiz_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Quiz session not found")

        questions = await db_client.get_questions_by_file(session["file_id"])

        if export_type == "results_with_answers":
            responses = await db_client.get_quiz_responses(session_id)
        else:
            responses = None

        # Generate export file
        file_path = exporter.export(
            questions=questions,
            responses=responses,
            export_type=export_type,
            file_format=file_format,
            session_id=session_id
        )

        # Record export
        await db_client.create_export_record(
            session_id=session_id,
            export_type=export_type,
            file_format=file_format
        )

        return FileResponse(
            file_path,
            media_type="application/pdf" if file_format == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=os.path.basename(file_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/api/files")
async def list_files(user_id: Optional[str] = None):
    """List all uploaded files"""
    try:
        files = await db_client.get_all_files(user_id)
        return JSONResponse({
            "success": True,
            "files": files
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch files: {str(e)}")


@app.get("/api/questions/{file_id}")
async def get_questions(file_id: str):
    """Get all questions for a file"""
    try:
        questions = await db_client.get_questions_by_file(file_id)
        return JSONResponse({
            "success": True,
            "questions": questions
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch questions: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    print(f"""
    ╔═══════════════════════════════════════════════════════╗
    ║   Smart AI MCQ Generator - Backend Server            ║
    ║   Running on http://{host}:{port}                    ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )