from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scenario_pipeline.adapters.waymo import convert_waymo_scenario, convert_waymo_tfrecord
from scenario_pipeline.tests.fake_waymo import build_scenario
from scenario_pipeline.validation import validate_scenario


class WaymoAdapterTest(unittest.TestCase):
    def test_proto_shaped_scenario(self) -> None:
        scenario = convert_waymo_scenario(build_scenario())
        self.assertEqual(scenario.timing.num_steps, 91)
        self.assertEqual(scenario.timing.anchor_index, 10)
        self.assertEqual(len(scenario.agents), 2)
        self.assertAlmostEqual(scenario.agents[0].x[0], 0.0)
        self.assertEqual(scenario.agents[1].type, "other")
        self.assertTrue(scenario.agents[1].roles["is_object_of_interest"])
        self.assertEqual(len(scenario.map_features), 2)
        self.assertEqual(scenario.map_features[0].successor_ids, ["101"])
        self.assertEqual(scenario.traffic_lights[0].states[0], "stop")
        self.assertEqual(scenario.traffic_lights[0].states[-1], "go")
        self.assertFalse(scenario.route.available)
        validate_scenario(scenario)

    def test_empty_map_and_missing_traffic_lights(self) -> None:
        proto = build_scenario()
        proto.map_features = []
        proto.dynamic_map_states = []
        scenario = convert_waymo_scenario(proto)
        self.assertFalse(scenario.quality.map_available)
        self.assertEqual(scenario.map_features, [])
        self.assertEqual(scenario.traffic_lights, [])
        validate_scenario(scenario)

    def test_real_tfrecord_when_waymo_environment_is_available(self) -> None:
        try:
            import tensorflow as tf
            from waymo_open_dataset.protos import scenario_pb2
        except ImportError:
            self.skipTest("TensorFlow/Waymo environment is not installed")

        proto = scenario_pb2.Scenario()
        proto.scenario_id = "minimal-tfrecord"
        proto.current_time_index = 0
        proto.sdc_track_index = 0
        for index in range(2):
            proto.timestamps_seconds.append(index * 0.1)
            proto.dynamic_map_states.add()
            state = proto.tracks.add().states.add() if index == 0 else proto.tracks[0].states.add()
            state.center_x = float(index)
            state.length = 4.8
            state.width = 2.0
            state.height = 1.6
            state.velocity_x = 10.0
            state.valid = True
        proto.tracks[0].id = 1
        proto.tracks[0].object_type = 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimal.tfrecord"
            with tf.io.TFRecordWriter(str(path)) as writer:
                writer.write(proto.SerializeToString())
            converted = convert_waymo_tfrecord(path)
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].timing.num_steps, 2)


if __name__ == "__main__":
    unittest.main()
