// Copy buttons for code blocks (progressive enhancement; no dependencies).
document.querySelectorAll(".copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const card = btn.closest(".code-card");
    const code = card ? card.querySelector("code") : null;
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code.innerText);
      const label = btn.textContent;
      btn.textContent = "copied";
      setTimeout(() => { btn.textContent = label; }, 1400);
    } catch {
      // Clipboard API unavailable (e.g. non-secure context): select instead.
      const range = document.createRange();
      range.selectNodeContents(code);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  });
});
