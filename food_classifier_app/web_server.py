"""Script to host a web server which returns predictions of food types given an image.

required packages: uvicorn, starlette, aiohttp, python-multipart

Based on:
* Starlette: https://www.starlette.io/applications/
* fast.ai lecture 2 - "download notebook": https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb
* cougar-or-not: https://github.com/simonw/cougar-or-not

The fastai learner model used was generated with ../notebooks/food_classifier.ipynb.
"""
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import aiohttp
from io import BytesIO

from fastai.vision import load_learner, torch, open_image

# this my need to be adjusted to contain your learner
learner = {"dir": ".", "fname": "export.pkl", "model": None, "i2c": None}

# =========== Helper functions ===========


def startup():
    print("Ready to go wild!")


async def get_bytes(url):
    # Collects the bytes for a single image from a dedicated URL
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def predict_image_from_bytes(_bytes):
    """Function to perform food predictions given some image.

    """

    img = open_image(BytesIO(_bytes))
    pred_class, pred_idx, probs = learner["model"].predict(img)
    topk = 3

    topk = len(probs) if topk > len(probs) else topk
    top_idx = torch.topk(probs, topk).indices

    return JSONResponse({
        "predictions": sorted([(learner["i2c"][_ix], float(probs[_ix])) for _ix in top_idx.numpy()],
                              key=lambda x: x[1], reverse=True)
    })

# =========== Web server functions ===========


app = Starlette(on_startup=[startup])


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    _bytes = await data["file"].read()
    return predict_image_from_bytes(_bytes)


@app.route("/classify_url", methods=["GET"])
async def classify_url(request):
    _bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(_bytes)


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>        
        """
    )


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


def main():
    # loading an already trained learner
    learner["model"] = load_learner(path=learner["dir"], file=learner["fname"])
    # storing index -> label maps
    learner["i2c"] = {i: c for c, i in learner["model"].data.c2i.items()}

    # starting a local web server
    host = "0.0.0.0"
    port = 8000
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()