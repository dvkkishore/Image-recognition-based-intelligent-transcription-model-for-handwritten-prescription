import { useEffect, useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  async function handleSubmit(event) {
    event.preventDefault();
    setResult('');
    setError('');

    if (!file || !['image/png', 'image/jpeg'].includes(file.type)) {
      setError('Select a PNG or JPEG image.');
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setError('Please select an image smaller than 5 MB.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/upload`, formData);
      setResult(response.data.prediction ?? '');
    } catch (uploadError) {
      setError(uploadError.response?.data?.error || 'Transcription failed. Check that the local API is running.');
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(event) {
    setFile(event.target.files?.[0] ?? null);
    setResult('');
    setError('');
  }

  return (
    <main className="w-full min-h-[80vh] flex items-center justify-center flex-col p-4 gap-4">
      <h1 className="font-bold text-xl">Experimental handwriting OCR</h1>
      <p className="max-w-lg text-center text-sm">
        This IAM handwriting demo is not validated for reading medical prescriptions.
        Do not use its output to make medication decisions.
      </p>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4 items-center">
        <input
          aria-label="Handwriting image"
          className="border"
          type="file"
          accept="image/png,image/jpeg"
          onChange={handleFileChange}
          required
        />
        <button
          type="submit"
          className="w-[10rem] h-[3rem] border-black border-2 rounded-lg"
          disabled={loading}
        >
          {loading ? 'Processing…' : 'Transcribe'}
        </button>
      </form>
      {preview && <img src={preview} alt="Uploaded handwriting preview" className="max-w-md max-h-72 object-contain" />}
      {error && <p role="alert" className="text-red-700">{error}</p>}
      {result && <p className="font-bold text-xl">Transcription: {result}</p>}
    </main>
  );
}

export default App;
