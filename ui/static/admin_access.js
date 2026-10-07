document.querySelectorAll("[data-logout]").forEach(button => button.addEventListener("click", async () => {
  const response = await fetch("/api/auth/logout", {method: "POST"});
  if (response.ok) location.assign("/");
}));

const accountForm = document.getElementById("createAccountForm");
if (accountForm) {
  const table = document.getElementById("accountTable");
  const message = document.getElementById("accountMessage");
  async function loadAccounts() {
    const response = await fetch("/api/admin/users");
    if (!response.ok) throw new Error("Could not load accounts.");
    const {users} = await response.json();
    table.replaceChildren();
    const head = table.createTHead().insertRow();
    ["Email", "Role", "Sign in", "Status", "Action"].forEach(label => { const th = document.createElement("th"); th.textContent = label; head.append(th); });
    const body = table.createTBody();
    for (const user of users) {
      const row = body.insertRow();
      [user.email, user.role.replace("_", " "), [user.has_password && "Password", user.google_connected && "Google"].filter(Boolean).join(", ") || "Google eligible", user.active ? "Active" : "Revoked"].forEach(value => { const cell = row.insertCell(); cell.textContent = value; });
      const cell = row.insertCell();
      if (user.role === "admin") {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = user.active ? "Revoke access" : "Restore access";
        button.addEventListener("click", async () => {
          message.textContent = "Updating access…";
          try {
            const response = await fetch(`/api/admin/users/${user.id}/access`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({active: !user.active})});
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || "Access update failed.");
            message.textContent = `${user.email} access ${user.active ? "revoked" : "restored"}.`;
            await loadAccounts();
          } catch (error) { message.textContent = error.message; }
        });
        cell.append(button);
      }
    }
  }
  accountForm.addEventListener("submit", async event => {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(accountForm));
    message.textContent = "Creating account…";
    try {
      const response = await fetch("/api/admin/users", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Account creation failed.");
      accountForm.reset();
      message.textContent = `Created ${result.role.replace("_", " ")} account for ${result.email}.`;
      await loadAccounts();
    } catch (error) { message.textContent = error.message; }
  });
  loadAccounts().catch(error => { message.textContent = error.message; });
}
