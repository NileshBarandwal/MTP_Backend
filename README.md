# MTP_Backend

*  Model versioning works
*  Input and model hashes are embedded in the attestation
*  Output is signed with the private key
*  Public key is retrievable
*  Client can verify everything using a simple script

---

## Overview of Backend Components

### 📁 Directory Structure (backend only)

```
mtp/
├── inference_server.py          # Flask server with attestation + version support
├── generate_keys.py             # To generate signing keypair
├── models/
│   ├── mnist_v1.h5
│   ├── mnist_v2.h5
├── keys/
│   ├── private_key.pem
│   └── public_key.pem
```

---

## Step 1: Key Generation (Run once)

### `generate_keys.py`

Run this:

```bash
python3 generate_keys.py
```

---

## Step 2: Flask Server (Core API)

### `inference_server.py`

---

## Step 3: Testing the API (from client)

```bash
curl -X POST -F "file=@digit1.png" -F "model_version=v1" http://localhost:5000/infer -o attestation_digit1.json
```

---

## Step 4: Verifier Script (Client-Side)

### `verify_signature.py`

## Test the Full Flow

1. **Start the server:**

```bash
python3 inference_server.py
```

2. **Run inference:**

```bash
curl -X POST -F "file=@digit1.png" -F "model_version=v1" http://localhost:5000/infer -o attestation_digit1.json
```

3. **Download public key:**

```bash
curl http://localhost:5000/get-public-key -o public_key.pem
```

4. **Verify:**

```bash
python3 verify_signature.py attestation_digit1.json public_key.pem
```
