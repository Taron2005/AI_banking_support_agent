import React from "react";

const URL_SPLIT = /(https?:\/\/[^\s)\]>"']+)/gi;

function trimUrlPunctuation(url) {
  return url.replace(/[.,;։)\]»'"`]+$/u, "");
}

/**
 * Renders plain text with clickable URLs (assistant answers often include source links).
 */
export function MessageContent({ text, className }) {
  if (text == null || text === "") return null;
  const s = String(text);
  const parts = s.split(URL_SPLIT);
  return (
    <div className={className || "message-text"}>
      {parts.map((part, i) => {
        if (part.match(/^https?:\/\//i)) {
          const href = trimUrlPunctuation(part);
          return (
            <a key={i} href={href} target="_blank" rel="noreferrer" className="message-link">
              {part}
            </a>
          );
        }
        return <React.Fragment key={i}>{part}</React.Fragment>;
      })}
    </div>
  );
}
