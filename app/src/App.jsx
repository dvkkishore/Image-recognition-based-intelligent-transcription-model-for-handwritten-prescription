import React, { useState, useEffect } from 'react';
import axios from 'axios';

const App = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState();
  const [preview, setPreview] = useState(null);
  

  useEffect(() => {
    console.log('Result updated:', result);
  }, [result]); // Ensure 'result' is included in the dependency array

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', file);
    const objectUrl = URL.createObjectURL(file)
    setPreview(objectUrl)

    console.log('Form submitted');

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log(response.status)
      if (response.status === 200) {
        setResult(response.data.prediction);
        console.log('Upload successful');
      } else {
        console.log('Upload failed');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      console.log('Upload failed');
    }
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  return (
    <div className='w-full h-[80vh] flex items-center justify-center flex-col' >
      <form onSubmit={handleSubmit} className='flex flex-col gap-4 items-center '>
        <input className='border flex flex-col'
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button type="submit" className=' w-[10rem] h-[3rem] border-black border-2 rounded-lg'>Upload</button>
      </form>
      {preview && <div><img src={preview} ></img></div>}
      {result && <div className=' font-bold text-[2rem]'> Transcription : {result}</div>}
    </div>
  );
};

export default App;
