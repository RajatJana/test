'use client';

import React, { useState, useRef, useEffect } from 'react';
import { X, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea, ScrollBar } from '../ui/scroll-area';
import { UploadIcon } from '../icons/appIcons';
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
  const [generalFiles, setGeneralFiles] = useState<File[]>([]);
  const [yamlFile, setYamlFile] = useState<File | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [showSuccessMessage, setShowSuccessMessage] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [extractedPayload, setExtractedPayload] = useState<any>(null);
  const generalInputRef = useRef<HTMLInputElement>(null);
  const yamlInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (type: 'general' | 'yaml', e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      if (type === 'general') {
        setGeneralFiles(Array.from(e.target.files));
      } else {
        setYamlFile(e.target.files[0]); // Single YAML file
      }
      onUploadComplete([]); // Reset extracted data
      onExtractTriggered();
    }
  };

  const handleClearFile = (type: 'general' | 'yaml', index?: number) => {
    if (type === 'general') {
      if (index !== undefined) {
        setGeneralFiles((prev) => prev.filter((_, i) => i !== index));
      } else {
        setGeneralFiles([]);
        if (generalInputRef.current) generalInputRef.current.value = '';
      }
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
      setGeneralFiles([]);
      if (generalInputRef.current) generalInputRef.current.value = '';
    }
    onUploadComplete([]);
    onExtractTriggered();
  };

  const formatSize = (bytes: number) => `${(bytes / 1024).toFixed(1)}Kb`;

  const handleExtract = async () => {
    if (!generalFiles.length && !yamlFile) return;
    setIsExtracting(true);
    onExtractTriggered();

    try {
      const formData = new FormData();
      if (uploadType === 'document') {
        generalFiles.forEach((file) => formData.append('files', file));
      } else if (uploadType === 'yaml' && yamlFile) {
        formData.append('yaml', yamlFile);
      }

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
      <div className="bg-[#ececec] dark:bg-[#181818] rounded-lg shadow w-full text-white p-2 h-[250px]">
        <p className="text-black dark:text-white text-sm font-semibold">Upload Files</p>
        <div className="mt-[10px] border border-[#535458] dark:border-[#535458] rounded-lg p-5 grid grid-cols-[1fr_auto_1fr] gap-4">
          {/* Document Files Upload Section */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <input
                type="radio"
                id="document-radio"
                name="upload-type"
                value="document"
                checked={uploadType === 'document'}
                onChange={() => handleUploadTypeChange('document')}
                disabled={isExtracting}
              />
              <label htmlFor="document-radio" className="text-black dark:text-white text-sm">
                Select BRD
              </label>
            </div>
            <div className="flex items-center gap-2">
              <input
                ref={generalInputRef}
                type="file"
                accept=".pdf,.csv,.docx"
                multiple
                onChange={(e) => handleFileChange('general', e)}
                className="hidden"
                id="general-upload"
                disabled={isExtracting || uploadType !== 'document'}
              />
              <Button
                onClick={() => generalInputRef.current?.click()}
                className={`w-[10vw] bg-[#E5E5E5] dark:bg-[#E5E5E5] text-black dark:text-black hover:bg-[#D0D0D0] dark:hover:bg-[#D0D0D0] border border-[#535458] rounded-md py-1.5 px-3 ${isExtracting || uploadType !== 'document' ? 'opacity-50 cursor-not-allowed' : ''}`}
                disabled={isExtracting || uploadType !== 'document'}
              >
                Choose Files
              </Button>
              <div className="flex items-center gap-1">
                {generalFiles.length === 0 ? (
                  <p className="text-[#7C7F84] dark:text-[#7C7F84] text-sm">No files chosen</p>
                ) : (
                  <p className="text-[#7C7F84] dark:text-[#7C7F84] text-sm">
                    {generalFiles.length} file(s) selected
                  </p>
                )}
              </div>
            </div>
            {generalFiles.length > 0 && (
              <div className="mt-1 max-h-20 overflow-y-auto border border-[#535458] rounded-md p-1">
                {generalFiles.map((file, index) => (
                  <div key={`${file.name}-${index}`} className="flex justify-between items-center py-0.5 text-xs text-black dark:text-white">
                    <span className="truncate max-w-[150px]">{file.name} ({formatSize(file.size)})</span>
                    <Button
                      onClick={() => handleClearFile('general', index)}
                      className="p-0 bg-transparent hover:bg-transparent min-w-0 ml-2"
                      title="Clear file"
                      size="sm"
                    >
                      <X className="w-3 h-3 text-[#7C7F84]" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
            <p className="text-black dark:text-white text-xs font-semibold">Supported format</p>
            <p className="text-black dark:text-white text-xs">pdf, docx</p>
          </div>

          {/* Divider */}
          <div className="border-l border-[#535458] dark:border-[#535458]"></div>

          {/* YAML File Upload Section */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <input
                type="radio"
                id="yaml-radio"
                name="upload-type"
                value="yaml"
                checked={uploadType === 'yaml'}
                onChange={() => handleUploadTypeChange('yaml')}
                disabled={isExtracting}
              />
              <label htmlFor="yaml-radio" className="text-black dark:text-white text-sm">
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
                className={`w-[10vw] bg-[#E5E5E5] dark:bg-[#E5E5E5] text-black dark:text-black hover:bg-[#D0D0D0] dark:hover:bg-[#D0D0D0] border border-[#535458] rounded-md py-1.5 px-3 ${isExtracting || uploadType !== 'yaml' ? 'opacity-50 cursor-not-allowed' : ''}`}
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
              <div className="mt-1 max-h-20 overflow-y-auto border border-[#535458] rounded-md p-1">
                <div className="flex justify-between items-center py-0.5 text-xs text-black dark:text-white">
                  <span className="truncate max-w-[150px]">{yamlFile.name} ({formatSize(yamlFile.size)})</span>
                  <Button
                    onClick={() => handleClearFile('yaml')}
                    className="p-0 bg-transparent hover:bg-transparent min-w-0 ml-2"
                    title="Clear file"
                    size="sm"
                  >
                    <X className="w-3 h-3 text-[#7C7F84]" />
                  </Button>
                </div>
              </div>
            )}
            <p className="text-black dark:text-white text-xs font-semibold">Supported format</p>
            <p className="text-black dark:text-white text-xs">yaml</p>
          </div>
        </div>

        <Button
          onClick={handleExtract}
          disabled={(!generalFiles.length && !yamlFile) || isExtracting}
          className="text-xs h-10 mt-1 px-3 rounded-md"
        >
          Extract Requirements
        </Button>
      </div>

      {/* Fixed Card with Scrollable List & Footer */}
      <div className="relative bg-[#ececec] dark:bg-[#181818] rounded-lg shadow w-full text-white p-2 mt-4 h-[220px]">
        <h3
          className={`text-sm font-semibold ${
            extractedData.length === 0 || extractionStatus === 'failed'
              ? 'text-gray-700 dark:text-gray-500'
              : 'text-black dark:text-white'
          }`}
        >
          {extractedData.length === 0 || extractionStatus === 'failed'
            ? 'Extracted'
            : `Extracted ${extractedData.length} candidate requirements text with respective IDs`}
        </h3>

        <ScrollArea className="pr-1 mt-2 space-y-2 h-[179px] overflow-y-hidden border border-[#4D4B4B] bg-white dark:bg-[#26262675] rounded-lg">
          {extractedData.map((req, index) => (
            <div
              key={index}
              className="rounded-md px-4 py-2 text-sm border-b border-gray-600 last:border-none"
            >
              <span className="text-black dark:text-[#F5EBFF]">{req.requirement_id}</span>
              <p className="text-black dark:text-white">{req.requirement_text}</p>
              <div className="w-full space-y-2">
                <div className="mt-2 leading-relaxed text-black dark:text-white text-[13px] font-medium">
                  <span className="inline-block align-top bg-[#73ABB5] text-black dark:text-white text-[11px] px-2 py-1 rounded-md mr-2">
                    Text from BRD
                  </span>
                  <span className="inline">“{req.original_text}”</span>
                </div>
                <div className="flex flex-wrap gap-2 items-center justify-end w-full text-[11px] text-white">
                  <span className="rounded-md bg-[#344A2F] px-3 py-1 min-w-[75px]">
                    Page # {req.page_number}
                  </span>
                  <span className="text-white">{'>'}</span>
                  <span className="rounded-md bg-[#495A72] px-3 py-1 min-w-[110px] text-[11px]">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <p className="truncate">
                          Source Section:..<span className="text-[#00D7FF]">view more</span>
                        </p>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="text-[10px] max-w-[200px] break-words">{req.source_section}</p>
                      </TooltipContent>
                    </Tooltip>
                  </span>
                  <span className="text-white">{'>'}</span>
                  <span className="bg-[#655142] px-3 py-1 min-w-[75px] rounded-md">
                    Line # {req.line_number}
                  </span>
                </div>
              </div>
            </div>
          ))}
          <ScrollBar orientation="vertical"></ScrollBar>
        </ScrollArea>
      </div>

      {/* Footer message + button */}
      <div className="border-t border-gray-700 mt-2">
        <div
          className="flex rounded-lg justify-between items-center bg-[#f6f6f6] dark:bg-[#0D0D0D] h-[50px] py-4"
          style={{ marginTop: '1px' }}
        >
          {isExtracting ? (
            <span className="flex flex-row gap-1 ml-[10px] items-center">
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
          ) : extractionStatus === 'failed' ? (
            <span className="text-red-400">❌ Extraction Failed!</span>
          ) : showSuccessMessage ? (
            <span className="flex items-center gap-1 ml-[10px]">
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
            className={`${dynamicMargin} bg-black dark:bg-[#E5E5E5] text-white dark:text-black rounded-lg px-4 py-1`}
            disabled={isLoading || extractedData.length === 0 || extractionStatus === 'failed'}
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