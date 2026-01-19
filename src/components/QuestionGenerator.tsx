import { useState } from 'react';
import { Brain, Settings, Loader2 } from 'lucide-react';

interface QuestionGeneratorProps {
  fileId: string;
  fileName: string;
  onQuestionsGenerated: (questions: any[], sessionId: string) => void;
}

export function QuestionGenerator({
  fileId,
  fileName,
  onQuestionsGenerated,
}: QuestionGeneratorProps) {
  const [numQuestions, setNumQuestions] = useState(10);
  const [difficulty, setDifficulty] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setError(null);
    setProgress(0);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

      setProgress(20);
      const formData = new FormData();
      formData.append('file_id', fileId);
      formData.append('num_questions', numQuestions.toString());
      if (difficulty) {
        formData.append('difficulty', difficulty);
      }

      setProgress(40);
      const response = await fetch(`${apiUrl}/api/generate-questions`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Question generation failed');
      }

      setProgress(70);

      const sessionFormData = new FormData();
      sessionFormData.append('file_id', fileId);
      sessionFormData.append(
        'session_name',
        `Quiz - ${new Date().toLocaleString()}`
      );

      const sessionResponse = await fetch(`${apiUrl}/api/create-quiz-session`, {
        method: 'POST',
        body: sessionFormData,
      });

      const sessionData = await sessionResponse.json();

      if (!sessionResponse.ok) {
        throw new Error(sessionData.detail || 'Session creation failed');
      }

      setProgress(100);
      onQuestionsGenerated(sessionData.questions, sessionData.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="flex items-center space-x-3 mb-6">
          <Brain className="h-8 w-8 text-blue-600" />
          <h2 className="text-2xl font-bold text-gray-800">
            Generate MCQ Questions
          </h2>
        </div>

        <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <p className="text-sm text-blue-800">
            <span className="font-semibold">File:</span> {fileName}
          </p>
        </div>

        <div className="space-y-6">
          <div>
            <label className="flex items-center space-x-2 text-sm font-medium text-gray-700 mb-2">
              <Settings className="h-4 w-4" />
              <span>Number of Questions</span>
            </label>
            <input
              type="range"
              min="5"
              max="30"
              value={numQuestions}
              onChange={(e) => setNumQuestions(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              disabled={isGenerating}
            />
            <div className="flex justify-between text-sm text-gray-600 mt-2">
              <span>5</span>
              <span className="font-semibold text-blue-600">{numQuestions}</span>
              <span>30</span>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Difficulty Level (Optional)
            </label>
            <select
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isGenerating}
            >
              <option value="">Mixed (All Levels)</option>
              <option value="Easy">Easy</option>
              <option value="Medium">Medium</option>
              <option value="Hard">Hard</option>
            </select>
          </div>

          {isGenerating && (
            <div className="space-y-3">
              <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-sm text-gray-600 text-center">
                Generating questions using AI... This may take a minute.
              </p>
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className={`
              w-full py-3 px-6 rounded-lg font-medium text-white
              transition-all duration-300 transform
              ${
                isGenerating
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 hover:scale-105 active:scale-95'
              }
              flex items-center justify-center space-x-2
            `}
          >
            {isGenerating ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Brain className="h-5 w-5" />
                <span>Generate Questions with AI</span>
              </>
            )}
          </button>
        </div>

        <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            What happens next?
          </h3>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• AI analyzes your document content</li>
            <li>• Generates contextual MCQ questions</li>
            <li>• Creates plausible answer options</li>
            <li>• Provides explanations for each question</li>
          </ul>
        </div>
      </div>
    </div>
  );
}