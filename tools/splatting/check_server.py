"""Validate registration, host rendering and sequence HTTP delivery on a test server."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8197")
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    def get(path):
        with urllib.request.urlopen(args.server + path, timeout=30) as r:
            return json.load(r)
    info = get("/object_info")
    generator_nodes = {n for n, spec in info.items() if spec.get("category") == "SplatKit/4DAnyone"}
    sequence_nodes = {n for n, spec in info.items() if spec.get("category") == "SplatKit/Splatting"}
    assert generator_nodes == {"SplatKit_4DAnyone" + name for name in
                               ("ValidateInput", "ModelLoader", "GenerateViews", "PreviewGrid", "LoadView", "ExportFrameset")}, generator_nodes
    assert sequence_nodes == {"SplatKit_" + name for name in
                              ("SplatBackendSetup", "LoadFrameset", "TrainSequence", "LoadSequence",
                               "SequenceFrame", "SequencePreview", "SequenceInfo", "SequencePlayer")}, sequence_nodes
    assert "SplatKit_DatasetProject" in info
    prompt = {
        "1":{"class_type":"SplatKit_LoadSequence","inputs":{"folder":str(Path(args.sequence).resolve())}},
        "2":{"class_type":"SplatKit_SequencePlayer","inputs":{"sequence":["1",0]}},
        "3":{"class_type":"SplatKit_SequencePreview","inputs":{"sequence":["1",0],"width":128,"height":192,"every_nth":1}},
        "4":{"class_type":"PreviewImage","inputs":{"images":["3",0]}},
        "5":{"class_type":"SplatKit_SplatBackendSetup","inputs":{"confirm":False,"rebuild":""}},
    }
    request = urllib.request.Request(args.server + "/prompt", data=json.dumps({"prompt":prompt}).encode(),
                                     headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        queued = json.load(response)
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        history = get("/history/" + prompt_id)
        if prompt_id in history:
            result = history[prompt_id]
            break
        time.sleep(0.25)
    else:
        raise RuntimeError("Test prompt did not finish within 90 seconds")
    assert result["status"]["status_str"] == "success", result["status"]
    outputs = result["outputs"]
    assert len(outputs["4"]["images"]) == 2
    assert outputs["5"]["text"][0].startswith("Ready."), outputs["5"]
    token = outputs["2"]["splatkit"][0]["token"]
    index = get("/splatkit/seq/" + token + "/index.json")
    assert len(index["frames"]) == 2
    for name in index["frames"]:
        with urllib.request.urlopen(args.server + "/splatkit/seq/" + token + "/" + name) as response:
            assert response.read(8) == b"SPLATSH1"
    with urllib.request.urlopen(args.server + "/extensions/ComfyUI-SplatKit/player/index.html") as response:
        assert b"canvas" in response.read()
    print(json.dumps({"4danyone_nodes":len(generator_nodes),"splatting_nodes":len(sequence_nodes),"status":"success","preview_images":outputs["4"]["images"],
                      "player_url":args.server + "/extensions/ComfyUI-SplatKit/player/index.html?dir=/splatkit/seq/" + token + "/"},indent=2))

if __name__ == "__main__":
    main()
