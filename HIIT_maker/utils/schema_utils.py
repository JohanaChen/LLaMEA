HIIT_RESPONSE_SCHEMA = """{
  "summary": {
    "avg_hr": "<integer: average heart rate in bpm>",
    "max_hr": "<integer: peak heart rate in bpm>",
    "avg_power": "<float: average power in watts>",
    "max_power": "<float: peak power in watts>"
  },
  "per_interval_hr": [
    {
      "name": "<string: exercise or rest label>",
      "start_sec": "<integer>",
      "end_sec": "<integer>",
      "avg_hr": "<integer>",
      "peak_hr": "<integer>"
    }
  ],
  "per_interval_power": [
    {
      "name": "<string: exercise or rest label>",
      "start_sec": "<integer>",
      "end_sec": "<integer>",
      "avg_power": "<float>",
      "peak_power": "<float>"
    }
  ],
  "assumptions": "<string: brief physiological explanation>"
}"""

