const form = document.getElementById("unlockForm");
const message = document.getElementById("unlockMessage");
const params = new URLSearchParams(location.search);

document.getElementById("unlockNext").value = params.get("next") || "";

const errors = {
  invalid: "Invalid access token.",
  rate: "Too many attempts. Try again in five minutes.",
  request: "Invalid request.",
};
if (errors[params.get("token_error")]) {
  message.textContent = errors[params.get("token_error")];
}

form.addEventListener("submit", () => {
  const input = document.getElementById("accessToken");
  input.value = input.value.trim();
  message.textContent = "Checking token…";
});
