'use client';

import React, { useState, useRef, useEffect } from 'react';
import { X, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

export interface Requirement {
  requirement_id: string;
  requirement_text: string;
  original_text: string;
  source_section: string;
  page_number: number;
  line_number: number;
}

export interface UploadTabProps {
  onUploadComplete: (data: Requirement[]) => void;
  isUploadComplete: boolean;
  extractedData: Requirement[];
  extractionStatus: 'idle' | 'success' | 'failed';
  goToClassifier: (classifiedData: any[]) => void;
  onExtractTriggered: () => void;
  isSidebarCollapsed?: boolean;
}

const UploadTab = ({
  onUploadComplete,
  isUploadComplete,
  extractedData,
  extractionStatus,
  goToClassifier,
  onExtractTriggered,
  isSidebarCollapsed,
}: UploadTabProps) => {
  const [uploadType, setUploadType] = useState<'document' | 'yaml'>('document');
  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [yamlFile, setYamlFile] = useState<File | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [showSuccessMessage, setShowSuccessMessage] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [extractedPayload, setExtractedPayload] = useState<any>(null);
  const documentInputRef = useRef<HTMLInputElement>(null);
  const yamlInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (type: 'document' | 'yaml', e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      if (type === 'document') {
        setDocumentFile(e.target.files[0]);
      } else {
        setYamlFile(e.target.files[0]);
      }
      onUploadComplete([]);
      onExtractTriggered();
    }
  };

  const handleClearFile = (type: 'document' | 'yaml') => {
    if (type === 'document') {
      setDocumentFile(null);
      if (documentInputRef.current) documentInputRef.current.value = '';
    } else {
      setYamlFile(null);
      if (yamlInputRef.current) yamlInputRef.current.value = '';
    }
    onUploadComplete([]);
    onExtractTriggered();
  };

  const handleUploadTypeChange = (value: 'document' | 'yaml') => {
    setUploadType(value);
    if (value === 'document') {
      setYamlFile(null);
      if (yamlInputRef.current) yamlInputRef.current.value = '';
    } else {
      setDocumentFile(null);
      if (documentInputRef.current) documentInputRef.current.value = '';
    }
    onUploadComplete([]);
    onExtractTriggered();
  };

  const formatSize = (bytes: number) => `${(bytes / 1024).toFixed(1)}Kb`;

  const handleExtract = async () => {
    if (!documentFile && !yamlFile) return;
    
    setIsExtracting(true);
    onExtractTriggered();

    try {
      const formData = new FormData();
      
      if (uploadType === 'document' && documentFile) {
        formData.append('file', documentFile);
        
        const response = await fetch('http://127.0.0.1:8000/extract', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorText = await response.text();
          console.error('Server response:', errorText);
          throw new Error('Failed to extract requirements');
        }

        const data = await response.json();
        setExtractedPayload(data);
        onUploadComplete(data.extracted_requirements || []);
        setShowSuccessMessage(true);
      } else if (uploadType === 'yaml' && yamlFile) {
        console.log('YAML processing not yet implemented');
        setTimeout(() => {
          setShowSuccessMessage(true);
          onUploadComplete([]);
          setIsExtracting(false);
        }, 1000);
      }
    } catch (err) {
      console.error('Extraction error:', err);
      onUploadComplete([]);
    } finally {
      setIsExtracting(false);
    }
  };

  const handleClassification = async () => {
    if (!extractedPayload) return;
    setIsLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(extractedPayload),
      });

      if (!response.ok) throw new Error('Classification failed');

      const classifiedData = await response.json();
      goToClassifier(classifiedData);
    } catch (err) {
      console.error('Classification error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (showSuccessMessage) {
      const timer = setTimeout(() => setShowSuccessMessage(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [showSuccessMessage]);

  const dynamicMargin = isSidebarCollapsed ? 'ml-[500px]' : 'ml-[20px]';

  return (
    <>
      {/* Upload Section - Fixed height */}
      <div className="bg-[#ececec] dark:bg-[#181818] rounded-lg shadow w-full text-white p-3">
        <p className="text-black dark:text-white text-sm font-semibold mb-2">Upload Files</p>
        <div className="border border-[#535458] dark:border-[#535458] rounded-lg p-4 grid grid-cols-[1fr_auto_1fr] gap-4">
          {/* Document Files Upload Section */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <input
                type="radio"
                id="document-radio"
                name="upload-type"
                value="document"
                checked={uploadType === 'document'}
                onChange={() => handleUploadTypeChange('document')}
                disabled={isExtracting}
                className="cursor-pointer"
              />
              <label htmlFor="document-radio" className="text-black dark:text-white text-sm cursor-pointer">
                Select BRD
              </label>
            </div>
            <div className="flex items-center gap-2">
              <input
                ref={documentInputRef}
                type="file"
                accept=".pdf,.docx"
                onChange={(e) => handleFileChange('document', e)}
                className="hidden"
                id="document-upload"
                disabled={isExtracting || uploadType !== 'document'}
              />
              <Button
                onClick={() => documentInputRef.current?.click()}
                className={`w-[10vw] bg-[#E5E5E5] dark:bg-[#E5E5E5] text-black dark:text-black hover:bg-[#D0D0D0] dark:hover:bg-[#D0D0D0] border border-[#535458] rounded-md py-1.5 px-3 transition-all ${isExtracting || uploadType !== 'document' ? 'opacity-50 cursor-not-allowed' : ''}`}
                disabled={isExtracting || uploadType !== 'document'}
              >
                Choose File
              </Button>
              <div className="flex items-center gap-1">
                {!documentFile ? (
                  <p className="text-[#7C7F84] dark:text-[#7C7F84] text-sm">No file chosen</p>
                ) : (
                  <p className="text-[#7C7F84] dark:text-[#7C7F84] text-sm">1 file selected</p>
                )}
              </div>
            </div>
            {documentFile && (
              <div className="mt-1 border border-[#535458] rounded-md p-2 bg-white dark:bg-[#26262675] transition-all">
                <div className="flex justify-between items-center text-xs text-black dark:text-white">
                  <span className="truncate max-w-[150px]">{documentFile.name} ({formatSize(documentFile.size)})</span>
                  <Button
                    onClick={() => handleClearFile('document')}
                    className="p-0 bg-transparent hover:bg-red-500/10 min-w-0 ml-2 transition-colors"
                    title="Clear file"
                    size="sm"
                  >
                    <X className="w-3 h-3 text-[#7C7F84] hover:text-red-500" />
                  </Button>
                </div>
              </div>
            )}
            <div className="mt-1">
              <p className="text-black dark:text-white text-xs font-semibold">Supported format</p>
              <p className="text-black dark:text-white text-xs opacity-70">pdf, docx</p>
            </div>
          </div>

          {/* Divider */}
          <div className="border-l border-[#535458] dark:border-[#535458]"></div>

          {/* YAML File Upload Section */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <input
                type="radio"
                id="yaml-radio"
                name="upload-type"
                value="yaml"
                checked={uploadType === 'yaml'}
                onChange={() => handleUploadTypeChange('yaml')}
                disabled={isExtracting}
                className="cursor-pointer"
              />
              <label htmlFor="yaml-radio" className="text-black dark:text-white text-sm cursor-pointer">
                Select YAML
              </label>
            </div>
            <div className="flex items-center gap-2">
              <input
                ref={yamlInputRef}
                type="file"
                accept=".yaml,.yml"
                onChange={(e) => handleFileChange('yaml', e)}
                className="hidden"
                id="yaml-upload"
                disabled={isExtracting || uploadType !== 'yaml'}
              />
              <Button
                onClick={() => yamlInputRef.current?.click()}
                className={`w-[10vw] bg-[#E5E5E5] dark:bg-[#E5E5E5] text-black dark:text-black hover:bg-[#D0D0D0] dark:hover:bg-[#D0D0D0] border border-[#535458] rounded-md py-1.5 px-3 transition-all ${isExtracting || uploadType !== 'yaml' ? 'opacity-50 cursor-not-allowed' : ''}`}
                disabled={isExtracting || uploadType !== 'yaml'}
              >
                Choose File
              </Button>
              <div className="flex items-center gap-1">
                {!yamlFile ? (
                  <p className="text-[#7C7F84] dark:text-[#7C7F84] text-sm">No file chosen</p>
                ) : (
                  <p className="text-[#7C7F84] dark:text-[#7C7F84] text-sm">1 file selected</p>
                )}
              </div>
            </div>
            {yamlFile && (
              <div className="mt-1 border border-[#535458] rounded-md p-2 bg-white dark:bg-[#26262675] transition-all">
                <div className="flex justify-between items-center text-xs text-black dark:text-white">
                  <span className="truncate max-w-[150px]">{yamlFile.name} ({formatSize(yamlFile.size)})</span>
                  <Button
                    onClick={() => handleClearFile('yaml')}
                    className="p-0 bg-transparent hover:bg-red-500/10 min-w-0 ml-2 transition-colors"
                    title="Clear file"
                    size="sm"
                  >
                    <X className="w-3 h-3 text-[#7C7F84] hover:text-red-500" />
                  </Button>
                </div>
              </div>
            )}
            <div className="mt-1">
              <p className="text-black dark:text-white text-xs font-semibold">Supported format</p>
              <p className="text-black dark:text-white text-xs opacity-70">yaml</p>
            </div>
          </div>
        </div>

        {/* Extract Button - Always visible */}
        <div className="mt-3">
          <Button
            onClick={handleExtract}
            disabled={(!documentFile && !yamlFile) || isExtracting}
            className="text-sm h-10 px-4 rounded-md transition-all hover:scale-[1.02]"
          >
            {isExtracting ? (
              <span className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Extracting...
              </span>
            ) : (
              'Extract Requirements'
            )}
          </Button>
        </div>
      </div>

      {/* Requirements Display Section - Fixed height with scroll */}
      <div className="bg-[#ececec] dark:bg-[#181818] rounded-lg shadow w-full text-white p-3 mt-4 h-[280px]">
        <h3
          className={`text-sm font-semibold mb-2 transition-colors ${
            extractedData.length === 0
              ? 'text-gray-700 dark:text-gray-500'
              : 'text-black dark:text-white'
          }`}
        >
          {extractedData.length === 0
            ? 'Extracted Requirements'
            : `Extracted ${extractedData.length} candidate requirement${extractedData.length > 1 ? 's' : ''}`}
        </h3>

        <ScrollArea className="h-[calc(100%-32px)] border border-[#4D4B4B] bg-white dark:bg-[#26262675] rounded-lg">
          <div className="p-2 space-y-3">
            {extractedData.length === 0 ? (
              <div className="flex items-center justify-center h-32 text-gray-500 dark:text-gray-400">
                <p className="text-sm">No requirements extracted yet</p>
              </div>
            ) : (
                extractedData.map((req, index) => (
                  <div
                    key={index}
                    className="bg-[#f9f9f9] dark:bg-[#1a1a1a] rounded-lg p-4 border border-[#e0e0e0] dark:border-[#333] hover:border-[#73ABB5] dark:hover:border-[#73ABB5] transition-all shadow-sm hover:shadow-md"
                  >
                    {/* Requirement ID */}
                    <div className="mb-2">
                      <span className="text-sm font-semibold text-[#2563eb] dark:text-[#60a5fa] bg-[#dbeafe] dark:bg-[#1e3a5f] px-2 py-1 rounded">
                        {req.requirement_id}
                      </span>
                    </div>

                    {/* Requirement Text */}
                    <p className="text-sm text-black dark:text-white mb-3 leading-relaxed">
                      {req.requirement_text}
                    </p>

                    {/* Original Text Section */}
                    <div className="mb-3 p-3 bg-[#f0f9ff] dark:bg-[#0a2540] rounded-md border-l-4 border-[#73ABB5]">
                      <div className="flex items-start gap-2">
                        <span className="inline-flex items-center bg-[#73ABB5] text-black dark:text-white text-[10px] px-2 py-1 rounded font-medium whitespace-nowrap flex-shrink-0">
                          Text from BRD
                        </span>
                        <span className="text-[13px] text-black dark:text-white leading-relaxed">
                          "{req.original_text}"
                        </span>
                      </div>
                    </div>

                    {/* Metadata Tags */}
                    <div className="flex flex-wrap gap-2 items-center text-[11px]">
                      <span className="inline-flex items-center gap-1 rounded-md bg-[#344A2F] text-white px-3 py-1.5 font-medium">
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Page {req.page_number}
                      </span>
                      
                      <span className="text-gray-400">→</span>
                      
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="inline-flex items-center gap-1 rounded-md bg-[#495A72] text-white px-3 py-1.5 cursor-help hover:bg-[#5a6b83] transition-colors font-medium">
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            <span className="truncate max-w-[100px]">{req.source_section}</span>
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-[10px] max-w-[200px] break-words">{req.source_section}</p>
                        </TooltipContent>
                      </Tooltip>
                      
                      <span className="text-gray-400">→</span>
                      
                      <span className="inline-flex items-center gap-1 bg-[#655142] text-white px-3 py-1.5 rounded-md font-medium">
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                        </svg>
                        Line {req.line_number}
                      </span>
                    </div>
                  </div>
                ))
            )}
          </div>
          <ScrollBar orientation="vertical" />
        </ScrollArea>
      </div>

      {/* Footer - Fixed at bottom */}
      <div className="border-t border-gray-700 mt-2">
        <div
          className="flex rounded-lg justify-between items-center bg-[#f6f6f6] dark:bg-[#0D0D0D] h-[50px] px-4"
          style={{ marginTop: '1px' }}
        >
          {isExtracting ? (
            <span className="flex flex-row gap-2 items-center text-black dark:text-white">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 23.242 23.242"
                className="animate-spin"
              >
                <g id="bx-loader" transform="translate(-16 -16)">
                  <path
                    d="M16,26.459h5.811v2.324H16Zm17.432,0h5.811v2.324H33.432Zm-6.973,6.973h2.324v5.811H26.459Zm0-17.432h2.324v5.811H26.459Zm-7.877,4.225,1.643-1.643,4.109,4.109-1.643,1.643ZM36.66,35.017,35.017,36.66l-4.109-4.109,1.643-1.643ZM22.691,30.908l1.643,1.643L20.225,36.66l-1.643-1.643Zm8.216-8.217,4.109-4.108,1.643,1.644-4.109,4.108Z"
                    fill="#00c4ff"
                  />
                </g>
              </svg>
              Generating Extraction...
            </span>
          ) : showSuccessMessage ? (
            <span className="flex items-center gap-2 text-black dark:text-white animate-in fade-in duration-300">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 30.266 30.266"
              >
                <path
                  d="M30.266,15.133A15.133,15.133,0,1,1,15.133,0,15.133,15.133,0,0,1,30.266,15.133ZM22.756,9.4a1.419,1.419,0,0,0-2.043.042l-6.57,8.37-3.959-3.961a1.419,1.419,0,0,0-2.005,2.005l5.005,5.007a1.419,1.419,0,0,0,2.041-.038l7.551-9.439A1.419,1.419,0,0,0,22.758,9.4Z"
                  fill="#24d304"
                />
              </svg>
              Extracted Successfully!
            </span>
          ) : (
            <span></span>
          )}
          <Button
            className={`${dynamicMargin} bg-black dark:bg-[#E5E5E5] text-white dark:text-black rounded-lg px-6 py-2 transition-all hover:scale-[1.02]`}
            disabled={isLoading || extractedData.length === 0}
            onClick={handleClassification}
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <svg
                  className="animate-spin h-4 w-4 text-white dark:text-black"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                  />
                </svg>
                Classifying...
              </span>
            ) : (
              'Classify'
            )}
          </Button>
        </div>
      </div>
    </>
  );
};

export default UploadTab;