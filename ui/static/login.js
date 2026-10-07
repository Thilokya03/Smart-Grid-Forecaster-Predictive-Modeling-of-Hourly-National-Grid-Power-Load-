const message = document.getElementById("loginMessage");
const next = new URLSearchParams(location.search).get("next");

document.getElementById("continueWithToken").addEventListener("click", () => {
  signIn("/api/auth/login", {method: "gate"});
});

fetch("/api/auth/gate").then(response => response.json()).then(result => {
  if (!result.role) { location.reload(); return; }
  const firstSuperAdmin = result.bootstrap_available && result.role === "super_admin";
  document.getElementById("continueWithToken").hidden = !result.bootstrap_available || firstSuperAdmin;
  document.getElementById("bootstrapDivider").hidden = !result.bootstrap_available || firstSuperAdmin;
  if (firstSuperAdmin) {
    document.getElementById("loginIntro").textContent = "Create your first super admin account with email and password or with Google.";
    document.getElementById("emailSubmit").textContent = "Create account with email";
    document.getElementById("googleDividerText").textContent = "or create with Google";
    document.getElementById("loginFootnote").textContent = "This first account will receive super admin access.";
  }
});

async function signIn(path, body) {
  message.textContent = "Signing in…";
  try {
    const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Sign in failed.");
    const allowed = result.role === "super_admin" ? ["/admin", "/model-comparison", "/super-admin", "/super-admin/create-admin"] : ["/admin", "/model-comparison"];
    location.assign(allowed.includes(next) ? next : result.next);
  } catch (error) {
    message.textContent = error.message;
  }
}

document.getElementById("passwordForm").addEventListener("submit", event => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  signIn("/api/auth/login", {method: "password", email: data.get("email"), password: data.get("password")});
});

fetch("/api/auth/config").then(response => response.json()).then(config => {
  if (!config.google_client_id) {
    document.getElementById("googleUnavailable").hidden = false;
    return;
  }
  const script = document.createElement("script");
  script.src = "https://accounts.google.com/gsi/client";
  script.async = true;
  script.onload = () => {
    google.accounts.id.initialize({client_id: config.google_client_id, callback: result => signIn("/api/auth/google", {credential: result.credential})});
    google.accounts.id.renderButton(document.getElementById("googleButton"), {theme: "outline", size: "large", width: 330});
  };
  script.onerror = () => { document.getElementById("googleUnavailable").hidden = false; };
  document.head.appendChild(script);
}).catch(() => { document.getElementById("googleUnavailable").hidden = false; });
