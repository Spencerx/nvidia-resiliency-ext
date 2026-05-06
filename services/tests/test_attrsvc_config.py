# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from nvidia_resiliency_ext.services.attrsvc.config import parse_service_endpoint


def test_parse_service_endpoint_defaults_to_loopback():
    endpoint = parse_service_endpoint("")

    assert endpoint.host == "127.0.0.1"
    assert endpoint.port == 8000
    assert endpoint.display_url == "http://127.0.0.1:8000"


def test_parse_service_endpoint_accepts_explicit_bind_all_host():
    endpoint = parse_service_endpoint("", host="0.0.0.0")  # nosec B104

    assert endpoint.host == "0.0.0.0"
    assert endpoint.port == 8000
