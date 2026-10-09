from backend.tests.test_api import client, wait_analysis, wav_bytes


def test_feedback_training_updates_order_without_new_analysis(client):
    """真实解码的固定合成音频验证偏好完整数据路径，不代表实际用户标注。"""
    identities = []
    for index in range(15):
        frequency = 200 + (index // 3) * 1400 + index * 10
        response = client.post("/api/tracks", files={"file": (f"偏好测试{index}.wav", wav_bytes(frequency))})
        assert response.status_code == 201
        identities.append(response.json()["id"])
    before = {}
    for index, identity in enumerate(identities):
        track = wait_analysis(client, identity)
        before[identity] = track["analyzed_at"]
        assert client.put(f"/api/tracks/{identity}/feedback", json={"rating": index // 3 + 1, "factors": []}).status_code == 200
    jobs_before = client.get("/api/jobs").json()
    model = client.post("/api/profile/train").json()
    assert model["status"] == "ready", model
    ordered = client.get("/api/tracks?mode=personal&sort=score").json()
    assert [t["personal_score"] for t in ordered] == sorted(t["personal_score"] for t in ordered)
    assert all(t["analyzed_at"] == before[t["id"]] for t in ordered)
    assert client.get("/api/jobs").json() == jobs_before
