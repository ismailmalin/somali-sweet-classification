"""Quick smoke test for the two-stage prediction API."""
import urllib.request
import json

BOUNDARY = "boundary12345"

def call_predict(img_path):
    with open(img_path, "rb") as f:
        img_data = f.read()

    body = (
        b"--" + BOUNDARY.encode() + b"\r\n"
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + img_data + b"\r\n"
        b"--" + BOUNDARY.encode() + b"--\r\n"
    )

    req = urllib.request.Request(
        "http://localhost:8000/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

# Test 1: Non-sweet image (should be rejected at Stage 1)
print("=" * 60)
print("TEST 1: Non-sweet image (expect Stage 1 rejection)")
result = call_predict("data_non_sweet/not_somali_sweet/non_sweet_0001.jpg")
print(json.dumps(result, indent=2))

# Test 2: A halwo image (should pass Stage 1 and be classified in Stage 2)
import os
halwo_dir = "data/halwo"
halwo_img = os.path.join(halwo_dir, os.listdir(halwo_dir)[0])
print("\n" + "=" * 60)
print(f"TEST 2: Sweet image ({halwo_img}) — expect Stage 2 classification")
result2 = call_predict(halwo_img)
print(json.dumps(result2, indent=2))
