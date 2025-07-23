import json
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256

def verify_attestation(attestation_file, public_key_file):
    with open(attestation_file, "r") as f:
        data = json.load(f)

    att = data["attestation"]
    input_hash = data.get("input_hash")
    model_hash = att.get("model_hash")
    measurement = att.get("measurement")
    signature_hex = att.get("signature")

    # ✅ Step 1: Recompute measurement
    recomputed = SHA256.new((input_hash + model_hash).encode()).hexdigest()
    if recomputed != measurement:
        print("❌ Measurement mismatch! Integrity failure.")
        return

    # ✅ Step 2: Verify signature over the measurement string (not full attestation)
    h = SHA256.new(measurement.encode())
    signature = bytes.fromhex(signature_hex)

    with open(public_key_file, "rb") as f:
        key = RSA.import_key(f.read())

    try:
        pkcs1_15.new(key).verify(h, signature)
        print("✅ Signature is valid. Attestation trusted.")
    except (ValueError, TypeError):
        print("❌ Signature verification failed.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 verify_signature.py <attestation.json> <public_key.pem>")
    else:
        verify_attestation(sys.argv[1], sys.argv[2])

