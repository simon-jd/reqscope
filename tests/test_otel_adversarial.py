"""Independent rejection cases for ambiguous identity and exhausted sequences."""
import json
import unittest

from tests import test_otel as fixtures
from tests.test_otel import client_record, envelope, span
from fixtures.fake_engine.generate import A, frame_bytes, header


class OTelAdversarialTests(unittest.TestCase):
    setUp = fixtures.OTelImportTests.setUp
    call = fixtures.OTelImportTests.call
    def test_conflicting_explicit_identity_attribute_is_rejected(self):
        item = span()
        item["attributes"] += [{"key": "reqscope.internal_request_id", "value": {"stringValue": "a"}},
                               {"key": "reqscope.internal_request_id", "value": {"stringValue": "b"}}]
        self.input.write_text(json.dumps(envelope([item])))
        result = self.call()
        self.assertNotEqual(result.returncode, 0, "conflicting identity must not use the last attribute")
        self.assertFalse(self.output.exists())

    def test_same_external_id_cannot_bind_one_client_to_two_internal_requests(self):
        items = [span(), span()]
        for item, identity in zip(items, ("x-first", "x-second")):
            item["attributes"].append({"key": "reqscope.internal_request_id", "value": {"stringValue": identity}})
        self.input.write_text(json.dumps(envelope(items)))
        client = self.root / "client.json"
        client.write_text(json.dumps(client_record()))
        result = self.call("--client-log", str(client))
        self.assertNotEqual(result.returncode, 0, "client identity cannot depend on OTLP span order")
        self.assertFalse(self.output.exists())

    def test_import_cannot_wrap_exhausted_sequence_to_zero(self):
        self.source.write_bytes(b"RQSSEG01" + frame_bytes(header()) + frame_bytes({
            "type": "Session", "producer": A, "seq": 2**64 - 1, "state": "Finalized"}))
        self.input.write_text(json.dumps(envelope([span()])))
        result = self.call()
        self.assertNotEqual(result.returncode, 0, "sequence space exhausted; import must not wrap")
        self.assertFalse(self.output.exists())

