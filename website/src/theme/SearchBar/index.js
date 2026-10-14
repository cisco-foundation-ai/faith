import { useEffect, useRef } from "react";
import OriginalSearchBar from "@theme-original/SearchBar";

// The search plugin creates its icon-only clear button after typing and exposes
// no label prop. Keep this compatibility adjustment outside the dependency and
// preserve the plugin's clearing behavior. Return focus to the input before the
// activated clear button disappears, so keyboard navigation retains its place.
export default function SearchBar(props) {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    const labelClearButtons = () => {
      for (const button of container.querySelectorAll(
        'button[class*="searchClearButton"]',
      )) {
        button.setAttribute("aria-label", "Clear search");
        button.setAttribute("title", "Clear search");
        button.setAttribute("type", "button");
      }
    };
    labelClearButtons();
    const observer = new MutationObserver(labelClearButtons);
    observer.observe(container, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, []);

  const restoreInputFocus = (event) => {
    if (event.target.closest?.('button[class*="searchClearButton"]')) {
      containerRef.current.querySelector("input.navbar__search-input")?.focus();
    }
  };

  return (
    <div
      ref={containerRef}
      style={{ display: "contents" }}
      onClick={restoreInputFocus}
    >
      <OriginalSearchBar {...props} />
    </div>
  );
}
