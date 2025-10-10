'use client';

import { Button } from '@/components/ui/button';
import { useEffect, useState, type JSX } from 'react';
import PipelineFinaloutputChat from './finaloutputchat';
import { vscDarkPlus  } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { HTMLAttributes } from 'react';

type Props = {
  goToStep: (step: number) => void;
  finalPipeline: string ;
};


export default function Pipelinefinaloutput({ goToStep , finalPipeline}: Props): JSX.Element {
  const handleBack = () => goToStep(2);
  const [successMessage, setSuccessMessage] = useState<string | null>(
    'Extracted Final Output Successfully!'
  );

  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => {
        setSuccessMessage(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [successMessage]);

  const [showChat, setShowChat] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
const [ciCdTool, setCiCdTool] = useState<string | null>(null);


// const handleDownload = () => {
//   if (!finalPipeline) return;
//   const blob = new Blob([finalPipeline], { type: 'text/plain' });
//   const link = document.createElement('a');
//   link.href = URL.createObjectURL(blob);
//   link.download = 'Pipeline';
//   document.body.appendChild(link);
//   link.click();
//   document.body.removeChild(link);
// };
const handleDownload = () => {
  console.log('Final Pipeline:', finalPipeline);
  console.log(typeof finalPipeline);
  console.log('CI/CD Tool:', ciCdTool);
  if (!finalPipeline) return;
  console.log(finalPipeline)
  const tool = finalPipeline.toString().toLowerCase().includes('azure') ? 'azure' : finalPipeline.toString().toLowerCase().includes('gitlab') ? 'gitlab' : 'jenkins';
  console.log('Detected Tool:', tool);
  let fileName: string | null = null;

  if (tool === 'jenkins') {
    fileName = 'Jenkinsfile'; // no extension
  } else if (tool === 'azure') {
    fileName = 'azure-pipeline.yml'; // with .yml extension
  } else if (tool === 'gitlab') {
    fileName = '.gitlab-ci.yml';
  }

  if (!fileName) return; // prevent download if tool is unrecognized

  const blob = new Blob([finalPipeline], { type: 'text/plain' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

const detectedLang = finalPipeline.includes("pipeline {") ? "groovy" : "yaml";
const cleanedPipeline = finalPipeline
  .replace(/^```(?:\w+)?\n/, '') // remove starting ```
  .replace(/```$/, '')           // remove ending ```
  .trim();


  return (
    <div className="flex flex-col">
      <div className="flex flex-row space-x-2 mt-2">
        {/* LEFT SIDEBAR */}
        {isSidebarOpen && (
          <div className="bg-[#f9f9f9] dark:bg-[#171717] p-3 w-full rounded-lg shadow-md text-black dark:text-white text-[14px] space-y-4 h-[29rem] ">
            <p>Final Output</p>

            <div className="h-[25.5rem] rounded-md border border-gray-700 flex flex-col relative">
              <pre className="bg-[#f5f5f5] dark:bg-[#1f1f1f] text-black dark:text-white p-2 overflow-y-auto rounded-md text-sm whitespace-pre-wrap flex-grow">
                {/* {finalPipeline} */}
 {finalPipeline  &&(
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          code(props) {
            const { className, children } = props;
            const match = /language-(\w+)/.exec(className || '');
            const lang = match ? match[1] : detectedLang;

            return (
              <SyntaxHighlighter
                language={lang}
                style={vscDarkPlus}
                PreTag="div"
                customStyle={{
                  borderRadius: '10px',
                  fontSize: '0.95rem',
                  backgroundColor: '#1f1f1f',
                  color: 'white',
                }}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            );
          },
        }}
      >
        {`\`\`\`${detectedLang}\n${cleanedPipeline}\n\`\`\``}
      </ReactMarkdown>
    )}

              </pre> 
 

              <Button
    onClick={handleDownload}
    className="absolute top-2 right-2 h-7 px-1 py-1 text-sm bg-[#ececec] dark:bg-[#323233] text-sm text-black dark:text-white rounded-md flex items-center"
  >
    <img src="./download.svg" className="w-4 h-3 " alt="Download" />
    Download
  </Button>


              <div className="bg-[#fafafa] dark:bg-[#171717] h-[3rem] p-1 rounded-md flex items-center gap-2 text-black dark:text-white mt-auto px-4">
                <span className="text-sm italic">
                  Would you like to connect with the Assistant?
                </span>
                  <span
                    onClick={() => {
                      setShowChat(true);
                      setIsSidebarOpen(false);
                    }}
                    className="text-white dark:text-black text-center h-6 w-10 bg-[#171717] dark:bg-white rounded-md cursor-pointer"
                  >
                    Yes
                  </span>
                
              </div>
            </div>
          </div>
        ) }
        
  
        {!isSidebarOpen && (
  <div
    className="w-8 h-[29rem] rounded-md bg-[#f9f9f9] dark:bg-[#1F1F1F] text-sm border border-gray-700 flex items-center justify-center cursor-pointer"
    onClick={() => {
      setShowChat(false);
      setIsSidebarOpen(true);
    }}
  >
    <p className="text-black dark:text-white text-sm tracking-widest -rotate-90 whitespace-nowrap">
      Final Output {'>'}
    </p>
  </div>
)}


        {/* RIGHT CHAT AREA */}
        <div className="flex-1"
                  >{showChat && <PipelineFinaloutputChat goToStep={goToStep}
                  setIsSidebarOpen={setIsSidebarOpen} 
  setShowChat={setShowChat} 
  showChat={showChat} 
  finalPipeline ={finalPipeline} />}</div>
      </div>

      {/* Bottom Message + Button Area */}
      <div className="flex flex-row  items-center justify-between w-full">
        <div className="flex flex-row mt-1 text-sm space-x-2 items-center">
          {successMessage && (
            <>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="18"
                height="18"
                viewBox="0 0 30.266 30.266"
              >
                <path
                  d="M30.266,15.133A15.133,15.133,0,1,1,15.133,0,15.133,15.133,0,0,1,30.266,15.133ZM22.756,9.4a1.419,1.419,0,0,0-2.043.042l-6.57,8.37-3.959-3.961a1.419,1.419,0,0,0-2.005,2.005l5.005,5.007a1.419,1.419,0,0,0,2.041-.038l7.551-9.439A1.419,1.419,0,0,0,22.758,9.4Z"
                  fill="#24d304"
                />
              </svg>
              <span className="text-black dark:text-white">{successMessage}</span>
            </>
          )}
        </div>

        <div className="flex mt-1">
          <Button
            onClick={handleBack}
            className="px-4 py-2 mt-1 bg-[#2B2B2B] text-white rounded-md h-8 w-24"
          >
            Back
          </Button>
        </div>
      </div>
    </div>
  );
}
