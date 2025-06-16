{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "700c4ce6-9729-4838-aa91-53fa634e5b3e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-06-16T11:59:48.748214Z",
     "iopub.status.busy": "2025-06-16T11:59:48.747900Z",
     "iopub.status.idle": "2025-06-16T11:59:48.846011Z",
     "shell.execute_reply": "2025-06-16T11:59:48.845295Z",
     "shell.execute_reply.started": "2025-06-16T11:59:48.748187Z"
    }
   },
   "outputs": [],
   "source": [
    "import joblib\n",
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import json\n",
    "from io import StringIO\n",
    "\n",
    "def model_fn(model_dir):\n",
    "    model_path = os.path.join(model_dir, \"model.joblib\")\n",
    "    model = joblib.load(model_path)\n",
    "    return model\n",
    "\n",
    "def input_fn(request_body, request_content_type):\n",
    "    if request_content_type == \"text/csv\":\n",
    "        return pd.read_csv(StringIO(request_body))\n",
    "    elif request_content_type == \"application/json\":\n",
    "        data = json.loads(request_body)\n",
    "        return pd.DataFrame(data)\n",
    "    else:\n",
    "        raise ValueError(\"Unsupported content type: {}\".format(request_content_type))\n",
    "\n",
    "def predict_fn(input_data, model):\n",
    "    return model.predict(input_data)\n",
    "\n",
    "def output_fn(prediction, content_type):\n",
    "    if content_type == \"application/json\":\n",
    "        return json.dumps(prediction.tolist())\n",
    "    elif content_type == \"text/csv\":\n",
    "        return \",\".join(map(str, prediction.tolist()))\n",
    "    else:\n",
    "        raise ValueError(\"Unsupported content type: {}\".format(content_type))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42e4a7e1-8e0e-44c1-b592-c5d9930dcb01",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
