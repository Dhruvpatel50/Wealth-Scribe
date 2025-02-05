'use client';

import * as React from 'react';
import { MdOutlineFileUpload } from 'react-icons/md';
import { useEdgeStore } from '../lib/edgestore';
import { useRouter } from 'next/navigation';

export default function Page() {
    const [file, setFile] = React.useState<File | null>(null);
    const [uploadProgress, setUploadProgress] = React.useState<number | null>(null);
    const [uploadStatus, setUploadStatus] = React.useState<string | null>(null);
    const [dragging, setDragging] = React.useState(false);
    const [showGenerateButton, setShowGenerateButton] = React.useState(false);
    const [isFileUploaded, setIsFileUploaded] = React.useState(false);
    const [isUrl, setIsUrl] = React.useState<string | null>(null);
    const [isLoading, setIsLoading] = React.useState(false); // Added for loader
    const { edgestore } = useEdgeStore();
    const router = useRouter();

    const handleFileUpload = async () => {
        if (file) {
            try {
                setUploadStatus('Uploading...');
                const res = await edgestore.publicFiles.upload({
                    file,
                    onProgressChange: (progress) => {
                        setUploadProgress(progress);
                    },
                });
                setUploadStatus('Upload successful!');
                setShowGenerateButton(true);
                setIsFileUploaded(true);
                setIsUrl(res.url);
            } catch (error) {
                console.error('Upload failed:', error);
                setUploadStatus('Upload failed. Please try again.');
            }
        }
    };

    const handleGenerate = async () => {
        if (!router || !isUrl) return;
        setIsLoading(true); // Show loader
        try {
            const res = await fetch("https://flaskapi-cr32.onrender.com/extract_financial_data", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ pdf_url: isUrl }),
            });

            if (res.ok) {
                const data = await res.json();
                console.log(data);
                const jsonParam = encodeURIComponent(JSON.stringify(data));
                router.push(`/chart?data=${jsonParam}`);
            } else {
                const errorDetails = await res.text();
                console.error('Failed to fetch data:', res.statusText);
            }
        } catch (error) {
            console.error('Error processing file:', error);
        } finally {
            setIsLoading(false); // Hide loader
        }
    };


    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (isFileUploaded) return;
        const uploadedFile = e.target.files?.[0];
        if (uploadedFile) {
            if (uploadedFile.type === 'application/pdf') {
                setFile(uploadedFile);
                setUploadProgress(null);
                setUploadStatus(null);
                setShowGenerateButton(false);
            } else {
                alert('Only PDF files are allowed!');
                e.target.value = '';
            }
        }
    };

    const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        if (isFileUploaded) return;
        e.preventDefault();
        setDragging(true);
    };

    const handleDragLeave = () => {
        setDragging(false);
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        if (isFileUploaded) return;
        e.preventDefault();
        setDragging(false);
        const uploadedFile = e.dataTransfer.files[0];
        if (uploadedFile) {
            if (uploadedFile.type === 'application/pdf') {
                setFile(uploadedFile);
                setUploadProgress(null);
                setUploadStatus(null);
                setShowGenerateButton(false);
            } else {
                alert('Only PDF files are allowed!');
            }
        }
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-[#212121] px-4 sm:px-8 relative">
            {isLoading && (
                <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-12 w-12"></div>
                    <style jsx>{`
                    .loader {
                        border: 10px solid #e5e7eb; 
                        border-top: 10px solid #636466; 
                        border-radius: 50%;
                        width: 120px;
                        height: 120px;
                        animation: spin 2s linear infinite;
                    }
                        @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                        }
                    `}</style>
                </div>
            )}
            <div
                className={`w-full sm:w-96 h-56 bg-[#212121] rounded-lg border-2 ${dragging ? 'border-[#424242]' : 'border-dashed border-[#616161]'
                    } flex flex-col items-center justify-center text-[#e0e0e0] transition-all ${isFileUploaded ? 'pointer-events-none opacity-50' : ''
                    }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    id="file-input"
                    className="hidden"
                    onChange={handleFileChange}
                    accept="application/pdf"
                    disabled={isFileUploaded}
                />
                <label
                    htmlFor="file-input"
                    className={`cursor-pointer text-center flex flex-col items-center ${isFileUploaded ? 'pointer-events-none opacity-50' : ''
                        }`}
                >
                    <MdOutlineFileUpload className="text-4xl text-[#757575] mb-2" />
                    <div className="text-lg font-medium">
                        {file ? file.name : 'Choose files'}
                    </div>
                    {!file && <div className="text-sm text-[#9e9e9e]">or drag and drop files</div>}
                </label>
            </div>

            {file && (
                <div className="mt-4 text-[#e0e0e0] flex flex-col items-center">
                    <p>
                        File selected: <span className="font-medium">{file.name}</span>
                    </p>
                    {!showGenerateButton && (
                        <button
                            onClick={handleFileUpload}
                            className="mt-4 bg-[#424242] hover:bg-[#616161] text-[#e0e0e0] font-semibold py-2 px-4 rounded-lg w-full sm:w-auto"
                        >
                            Upload
                        </button>
                    )}
                </div>
            )}

            {uploadProgress !== null && (
                <div className="mt-4 w-full sm:w-96 bg-[#424242] rounded-full h-4 overflow-hidden">
                    <div
                        className="bg-[#757575] h-full"
                        style={{ width: `${uploadProgress}%` }}
                    ></div>
                </div>
            )}

            {uploadStatus && (
                <p
                    className={`mt-4 text-sm ${uploadStatus.includes('successful')
                        ? 'text-[#81c784]'
                        : 'text-[#e57373]'
                        }`}
                >
                    {uploadStatus}
                </p>
            )}

            {showGenerateButton && (
                <button
                    onClick={handleGenerate}
                    className="mt-4 bg-[#424242] hover:bg-[#616161] text-[#e0e0e0] font-semibold py-2 px-4 rounded-lg w-full sm:w-auto"
                >
                    Generate
                </button>
            )}

            <p className="mt-6 text-sm text-[#bdbdbd] text-center">
                By uploading your files or using our service, you agree with our{' '}
                <a href="#" className="underline text-[#e0e0e0]">
                    Terms of Service
                </a>{' '}
                and{' '}
                <a href="#" className="underline text-[#e0e0e0]">
                    Privacy Policy
                </a>.
            </p>
        </div>
    );
}
