const form = document.getElementById("unlockForm");
const message = document.getElementById("unlockMessage");
form.addEventListener("submit", async event => {
  event.preventDefault();
  message.textContent = "Checking token…";
  try {
    const response = await fetch("/api/auth/unlock", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({token: String(new FormData(form).get("token") || "").trim()}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Access token was not accepted.");
    location.assign("/login" + location.search);
  } catch (error) {
    message.textContent = error.message;
  }
});
