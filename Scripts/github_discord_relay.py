#!/usr/bin/env python3

import hashlib
import hmac
import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib import error, request


DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "").strip()
HOST = os.environ.get("GITHUB_DISCORD_HOST", "0.0.0.0")
PORT = int(os.environ.get("GITHUB_DISCORD_PORT", "8787"))


def verify_signature(raw_body: bytes, signature_header: Optional[str]) -> bool:
    if not GITHUB_WEBHOOK_SECRET:
        return False
    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = hmac.new(
        GITHUB_WEBHOOK_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    received = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, received)


def short_sha(value: str) -> str:
    return value[:7] if value else "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_compare_url(payload: dict[str, Any]) -> Optional[str]:
    compare = payload.get("compare")
    if isinstance(compare, str) and compare:
        return compare
    repository = payload.get("repository") or {}
    html_url = repository.get("html_url")
    after = payload.get("after")
    if html_url and after:
        return f"{html_url}/commit/{after}"
    return None


def build_push_message(payload: dict[str, Any]) -> dict[str, Any]:
    repository = payload.get("repository") or {}
    sender = payload.get("sender") or {}
    ref = payload.get("ref", "")
    if isinstance(ref, str) and ref.startswith("refs/heads/"):
        branch = ref[len("refs/heads/") :]
    elif isinstance(ref, str) and ref:
        branch = ref
    else:
        branch = "unknown"
    commits = payload.get("commits") or []
    head_commit = payload.get("head_commit") or {}
    compare_url = make_compare_url(payload)
    repo_name = repository.get("full_name") or repository.get("name") or "unknown repo"
    sender_name = sender.get("login") or "someone"
    after = payload.get("after", "")

    lines = []
    for commit in commits[:10]:
        commit_id = short_sha(commit.get("id", ""))
        message = (commit.get("message") or "").splitlines()[0].strip()
        author = ((commit.get("author") or {}).get("name") or "unknown author").strip()
        commit_url = ""
        if repository.get("html_url") and commit.get("id"):
            commit_url = "{}/commit/{}".format(repository["html_url"], commit["id"])
        else:
            commit_url = compare_url or ""
        if commit_url:
            lines.append(f"[`{commit_id}`]({commit_url}) {message} - {author}")
        else:
            lines.append(f"`{commit_id}` {message} - {author}")

    extra_count = max(len(commits) - 10, 0)
    if extra_count:
        lines.append(f"...and {extra_count} more commit(s).")

    title = f"{repo_name}: {len(commits)} commit(s) pushed to {branch}"
    description = "\n".join(lines) if lines else f"Latest commit: `{short_sha(after)}`"

    embed = {
        "title": title,
        "description": description,
        "color": 0x5865F2,
        "timestamp": utc_now_iso(),
        "fields": [
            {
                "name": "Pushed by",
                "value": sender_name,
                "inline": True,
            },
            {
                "name": "Latest commit",
                "value": short_sha(head_commit.get("id") or after),
                "inline": True,
            },
        ],
    }

    if compare_url:
        embed["url"] = compare_url

    return {
        "username": "GitHub Commit Relay",
        "embeds": [embed],
        "allowed_mentions": {"parse": []},
    }


def build_ping_message(payload: dict[str, Any]) -> dict[str, Any]:
    repository = payload.get("repository") or {}
    hook_id = payload.get("hook_id", "unknown")
    repo_name = repository.get("full_name") or repository.get("name") or "unknown repo"

    return {
        "username": "GitHub Commit Relay",
        "content": f"GitHub webhook connected for `{repo_name}`. Hook id: `{hook_id}`.",
        "allowed_mentions": {"parse": []},
    }


def send_to_discord(message: dict[str, Any]) -> None:
    if not DISCORD_WEBHOOK_URL:
        raise RuntimeError("DISCORD_WEBHOOK_URL is not set")

    body = json.dumps(message).encode("utf-8")
    req = request.Request(
        DISCORD_WEBHOOK_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        if response.status >= 300:
            raise RuntimeError(f"Discord webhook returned HTTP {response.status}")


class GitHubRelayHandler(BaseHTTPRequestHandler):
    server_version = "GitHubDiscordRelay/1.0"

    def do_GET(self) -> None:
        if self.path not in ("/", "/healthz"):
            self.send_error(404, "Not found")
            return

        body = json.dumps({"ok": True, "service": "github-discord-relay"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/github":
            self.send_error(404, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return

        raw_body = self.rfile.read(content_length)
        signature_header = self.headers.get("X-Hub-Signature-256")
        event = self.headers.get("X-GitHub-Event")

        if not verify_signature(raw_body, signature_header):
            self.send_error(401, "Invalid signature")
            return

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON payload")
            return

        try:
            if event == "ping":
                send_to_discord(build_ping_message(payload))
            elif event == "push":
                send_to_discord(build_push_message(payload))
            else:
                self.respond_json(200, {"ok": True, "ignored_event": event})
                return
        except error.HTTPError as exc:
            self.respond_json(502, {"ok": False, "error": f"Discord HTTP {exc.code}"})
            return
        except Exception as exc:  # noqa: BLE001
            self.respond_json(500, {"ok": False, "error": str(exc)})
            return

        self.respond_json(200, {"ok": True, "event": event})

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write(
            f"[{self.log_date_time_string()}] {self.address_string()} {format % args}\n"
        )

    def respond_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    if not DISCORD_WEBHOOK_URL:
        raise SystemExit("DISCORD_WEBHOOK_URL is required")
    if not GITHUB_WEBHOOK_SECRET:
        raise SystemExit("GITHUB_WEBHOOK_SECRET is required")

    server = ThreadingHTTPServer((HOST, PORT), GitHubRelayHandler)
    print(f"Listening on http://{HOST}:{PORT}/github")
    print("Health check available at /healthz")
    server.serve_forever()


if __name__ == "__main__":
    main()
