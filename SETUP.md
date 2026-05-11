# Setup Guide

Quick reference for getting `home-watcher` running end-to-end in your homelab.

## 1. Prerequisites

- UDM SE / UDM Pro / UNVR running UniFi Protect 5.x
- A local UniFi admin account (NOT a UI.com cloud account — won't work with API)
  - Settings → Admins → Add New Admin → "Restrict to local access only"
  - Permissions: Network Read + Protect Read (minimum)
- k3s/k8s cluster with Flux (or run image manually via docker)
- An NTFY topic (use `ntfy.sh/<random>` for v1, self-host later)

## 2. Configure secrets

In `clusters/homelab/apps/home-watcher/secret.enc.yaml`:

```bash
SOPS_AGE_KEY_FILE=~/.config/sops/age/keys.txt \
  sops clusters/homelab/apps/home-watcher/secret.enc.yaml
```

Fill in:
- `unifi_user`: local admin username
- `unifi_pass`: local admin password
- `ntfy_topic`: random string (e.g. `home-watcher-xyz123abc`)
- `ntfy_token`: empty unless self-hosted NTFY with auth

## 3. Configure family member MACs

Edit `clusters/homelab/apps/home-watcher/configmap.yaml`:

```yaml
members:
  Malin: aa:bb:cc:dd:ee:ff
  Anna: 11:22:33:44:55:66
```

Get MAC addresses from UniFi Network UI:
- Clients → click family member's phone → Details → MAC Address

## 4. Configure per-camera rules

In the same configmap:

```yaml
cameras:
  Entrance:
    alert_weight: 0.3           # base score added if person detected
    family_zone: false
    always_alert_objects: [vehicle]
  Ocean:
    alert_weight: 0.2
    family_zone: true
    always_alert_objects: []
```

**Camera names must match Protect's exact names** (case-sensitive, including special chars like Å).

## 5. Subscribe on phone

Install [NTFY app](https://ntfy.sh/) → subscribe to your topic from secret above.

## 6. Train family faces

After pod is `Running`:

```bash
# Get a snapshot from a real camera as training data:
curl -k -c /tmp/c.txt -X POST https://192.168.0.10/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"<USER>","password":"<PASS>"}'
curl -k -b /tmp/c.txt "https://192.168.0.10/proxy/protect/api/cameras/<ID>/snapshot?force=true" \
  -o malin1.jpg

# Upload to home-watcher (use phone-camera selfies as initial seeds, then improve with real snapshots)
for photo in malin*.jpg; do
  curl -F "photo=@$photo" http://192.168.1.250:30040/api/subjects/Malin/photos
done

# Test recognition with a fresh image
curl -F "photo=@test.jpg" http://192.168.1.250:30040/api/test-recognize
```

Repeat for each family member.

## 7. Tune

After 1 week of real-world events, check pod logs:

```bash
KUBECONFIG=~/.kube/homelab-config kubectl logs -n home-watcher -l app.kubernetes.io/name=home-watcher --tail=200
```

Look for:
- `decision` log lines with `score`, `reasons`, `matched`
- Camera names that get too many/few alerts → adjust `alert_weight`
- Family members not recognized → add more training photos

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Pod ImagePullBackOff | GHA build still running, or `ghcr.io/bloomerab/home-watcher` repo isn't public |
| `protect_login_failed` in logs | UniFi creds wrong, MFA enabled on account, or account is UI.com not local |
| WebSocket disconnects every minute | Verify the local admin has Protect access (not just Network) |
| `No face detected in photo` when training | Photo is too small, too dark, or face is at extreme angle. Try clearer photo. |
| All events alert (even family) | Tolerance too tight (default 0.6). Tune `FACE_TOLERANCE` env var lower. |
| Family triggers alerts | Add more training photos covering different lighting/angles |
