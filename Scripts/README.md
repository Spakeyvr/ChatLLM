# GitHub To Discord Relay

This folder contains a tiny webhook relay for GitHub push events.

## What it does

- receives GitHub `push` webhooks in real time
- verifies the webhook signature with your shared secret
- posts a formatted message to a Discord webhook

## Environment variables

- `DISCORD_WEBHOOK_URL`: your Discord incoming webhook URL
- `GITHUB_WEBHOOK_SECRET`: the same secret you configure in GitHub
- `GITHUB_DISCORD_HOST`: optional, defaults to `0.0.0.0`
- `GITHUB_DISCORD_PORT`: optional, defaults to `8787`

## Run it

```bash
cd /Users/nevioknogler/Desktop/ChatLLM
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
export GITHUB_WEBHOOK_SECRET="replace-me"
python3 Scripts/github_discord_relay.py
```

The GitHub webhook endpoint will be:

```text
http://YOUR-HOST:8787/github
```

For production use, put it behind HTTPS with a reverse proxy or tunnel.

## GitHub setup

1. Open your repository on GitHub.
2. Go to `Settings` -> `Webhooks` -> `Add webhook`.
3. Set `Payload URL` to your public `/github` endpoint.
4. Set `Content type` to `application/json`.
5. Set `Secret` to the same value as `GITHUB_WEBHOOK_SECRET`.
6. Choose `Let me select individual events`.
7. Enable `Pushes`.
8. Save the webhook.

GitHub will send a `ping` right away, and the relay forwards that to Discord so you can confirm it works.
