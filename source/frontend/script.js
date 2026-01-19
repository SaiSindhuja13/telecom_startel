function ask() {
  const input = document.getElementById("question");
  const responseBox = document.getElementById("response");
  const loader = document.getElementById("loader");
  const button = document.getElementById("askBtn");

  const q = input.value.trim();
  if (!q) return;

  // UI state: loading
  loader.classList.remove("hidden");
  responseBox.innerText = "";
  button.disabled = true;

  fetch("http://127.0.0.1:5000/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q })
  })
    .then(res => res.json())
    .then(data => {
      responseBox.innerText = data.answer;
    })
    .catch(err => {
      responseBox.innerText = "Something went wrong. Please try again.";
      console.error(err);
    })
    .finally(() => {
      loader.classList.add("hidden");
      button.disabled = false;
    });
}
