"""Run the normal Streamlit controls on the original, hash-checked data."""
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from test_electricity_ao import LastLabel


class StreamlitFlowTests(unittest.TestCase):
    def test_public_baselines_and_paired_ao_display(self):
        app = AppTest.from_file(str(Path(__file__).with_name("electricity_app.py")), default_timeout=90).run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        stored = app.session_state["electricity_result"][1]
        self.assertEqual(stored["test_intervals"], 1344)
        self.assertEqual(stored["models"]["previous_interval"]["correct"], 1162)
        self.assertEqual(stored["models"]["warmup_majority"]["correct"], 819)
        self.assertEqual(stored["models"]["previous_day"]["correct"], 918)
        self.assertEqual(stored["models"]["online_exact_pattern"]["correct"], 1133)
        self.assertEqual(len(app.dataframe), 1)
        app.selectbox[0].select(1)
        app.selectbox[1].select("AO and baselines")
        with patch("electricity_ao.create_model", return_value=LastLabel()):
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        paired = app.session_state["electricity_result"][1]
        self.assertEqual(len(paired["rows"]), 48)
        self.assertEqual(paired["models"]["previous_interval"]["correct"], paired["models"]["ao_online"]["correct"])
        self.assertEqual(len(app.table), 2)
        self.assertEqual(len(app.get("download_button")), 1)


if __name__ == "__main__":
    unittest.main()
