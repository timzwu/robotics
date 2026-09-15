"""Lambda Cloud VM control for the sim work. Key file: ~/robotics/claude-lambda.txt (never in the repo).
    python sim/tools/lambda_vm.py types                 # prices + availability (free)
    python sim/tools/lambda_vm.py launch [--type gpu_1x_a6000] [--region us-south-2]   # prints the estimate, asks y/N
    python sim/tools/lambda_vm.py status                # running instances, IPs
    python sim/tools/lambda_vm.py ssh                   # prints the ssh command
    python sim/tools/lambda_vm.py terminate             # terminates every instance on the account and confirms none left
Billing is per minute from launch until terminate; there is no stop state."""
import argparse, json, os, pathlib, socket, sys, time, urllib.request, base64


class ApiError(Exception):
    pass

API = "https://cloud.lambda.ai/api/v1"
KEY = pathlib.Path.home() / "robotics" / "claude-lambda.txt"
SSH_KEY_NAME, SSH_KEY_FILE = "claude-lambda", pathlib.Path.home() / ".ssh" / "lambda_ed25519"


def call(path, method="GET", body=None):
    if not KEY.exists():
        raise ApiError(f"key file {KEY} missing")
    key = KEY.read_text().strip()
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{key}:".encode()).decode())
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "curl/8.4.0")  # Cloudflare in front of the API rejects urllib's default agent (403 error 1010)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise ApiError(f"{e.code} on {method} {path}: {e.read().decode()[:400]}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise ApiError(f"{method} {path}: {e}")


def port_open(ip, port=22):
    try:
        with socket.create_connection((ip, port), timeout=3):
            return True
    except OSError:
        return False


def types(show=True):
    d = call("/instance-types")["data"]
    rows = sorted((v["instance_type"]["price_cents_per_hour"] / 100, n, [r["name"] for r in v["regions_with_capacity_available"]]) for n, v in d.items())
    if show:
        for p, n, rs in rows:
            if n.startswith("gpu_1x"):
                print(f"${p:5.2f}/h  {n:22s} available: {', '.join(rs) or 'none'}")
    return d


def launch(itype, region, hours):
    if not SSH_KEY_FILE.exists():
        sys.exit(f"{SSH_KEY_FILE} missing: the instance would be unreachable")
    if SSH_KEY_NAME not in [k["name"] for k in call("/ssh-keys")["data"]]:
        sys.exit(f"no SSH key named '{SSH_KEY_NAME}' on the Lambda account")
    d = types(show=False)[itype]
    price = d["instance_type"]["price_cents_per_hour"] / 100
    avail = [r["name"] for r in d["regions_with_capacity_available"]]
    if region not in avail:
        sys.exit(f"{itype} not available in {region}; available: {avail or 'none'}")
    print(f"Launch 1x {itype} in {region} at ${price:.2f}/h; a {hours} h session is about ${price * hours:.2f}. Billed until terminated.")
    if os.environ.get("LAMBDA_YES") != "1" and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit("not launched")
    r = call("/instance-operations/launch", "POST", {"region_name": region, "instance_type_name": itype, "ssh_key_names": [SSH_KEY_NAME], "name": "isaac-sim"})
    iid = r["data"]["instance_ids"][0]
    print("launched", iid, "waiting for an IP and sshd …")
    for _ in range(60):
        time.sleep(10)
        inst = call(f"/instances/{iid}")["data"]
        st = inst.get("status")
        if st in ("unhealthy", "preempted", "terminating", "terminated"):
            sys.exit(f"instance {iid} is {st}: run `terminate`, then relaunch")
        if inst.get("ip") and st == "active" and port_open(inst["ip"]):
            print(f"active: ssh -i {SSH_KEY_FILE} -o StrictHostKeyChecking=accept-new ubuntu@{inst['ip']}")
            return
        print("  status:", st, inst.get("ip"))
    sys.exit(f"no ssh after 10 min but {iid} IS BILLING: run `python sim/tools/lambda_vm.py terminate`")


def status():
    data = call("/instances")["data"]
    if not data:
        print("no instances running"); return
    for i in data:
        print(i["id"], i["instance_type"]["name"], i["region"]["name"], i["status"], i.get("ip"), f"${i['instance_type']['price_cents_per_hour']/100:.2f}/h")


def ssh():
    data = call("/instances")["data"]
    if not data:
        sys.exit("no instances running")
    ip = data[0].get("ip")
    print(f"ssh -i {SSH_KEY_FILE} -o StrictHostKeyChecking=accept-new ubuntu@{ip}" if ip else "no ip yet (still booting)")


def terminate():
    ids = [i["id"] for i in call("/instances")["data"]]
    if not ids:
        print("nothing to terminate"); return
    r = call("/instance-operations/terminate", "POST", {"instance_ids": ids})
    print("terminate accepted for", [t["id"] for t in r["data"]["terminated_instances"]])
    for _ in range(36):
        time.sleep(5)
        try:
            left = [(i["id"], i["status"]) for i in call("/instances")["data"]]
        except ApiError as e:
            print("  poll failed, retrying:", e); continue
        if not left:
            print("terminated, no instances left"); return
        print("  still listed:", left)
    sys.exit("NOT CONFIRMED after 3 min: run `status` and check https://cloud.lambda.ai/instances")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["types", "launch", "status", "ssh", "terminate"])
    ap.add_argument("--type", default="gpu_1x_a6000"); ap.add_argument("--region", default="us-south-2"); ap.add_argument("--hours", type=float, default=2)
    a = ap.parse_args()
    try:
        {"types": types, "launch": lambda: launch(a.type, a.region, a.hours), "status": status, "ssh": ssh, "terminate": terminate}[a.cmd]()
    except ApiError as e:
        sys.exit(f"Lambda API: {e}")
