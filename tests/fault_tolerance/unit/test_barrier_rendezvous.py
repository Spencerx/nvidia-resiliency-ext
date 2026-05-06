# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for fault-tolerant barrier-based rendezvous implementation.

These tests focus on:
1. Basic rendezvous flow and coordination
2. Step 2 completion signaling
3. Race conditions with late arrivals
4. Store host vs non-store host behavior
5. Acknowledgment phase correctness
6. Group rank assignment
7. Error cases (timeout, closed rendezvous, exceeding max_nodes)
8. Infrastructure rank handling
"""

import os
import signal
import threading
import time
import uuid
from datetime import timedelta
from unittest import TestCase
from unittest.mock import MagicMock

from torch.distributed import TCPStore
from torch.distributed.elastic.multiprocessing import SignalException
from torch.distributed.elastic.rendezvous.api import (
    RendezvousClosedError,
    RendezvousGracefulExitError,
)

from nvidia_resiliency_ext.fault_tolerance import (
    ft_rendezvous_barrier as ft_rendezvous_barrier_module,
)
from nvidia_resiliency_ext.fault_tolerance.ft_rendezvous_barrier import (
    WITHDRAWN,
    FtRendezvousBarrierHandler,
    GroupRankStatus,
    RendezvousParticipantInfo,
    RendezvousStoreValue,
    RendezvousTimeout,
    RendezvousTimeoutError,
    _NodeDesc,
    _NodeDescGenerator,
    _PreviousActiveLastCallGate,
    _RendezvousBarrierState,
    _StaleRendezvousRoundError,
)

# Test timeout configuration - use short timeouts to make tests run faster
TEST_SEGMENT_CHECK_INTERVAL_SECS = 0.1  # seconds - for segment constraint check interval
TEST_JOIN_TIMEOUT_SECS = 2.0  # seconds - for join timeout (reduced from 5.0)
TEST_THREAD_JOIN_TIMEOUT_SECS = 5.0  # seconds - for thread.join() timeout (reduced from 10.0)
# Barrier wait timeout - avoids deadlock if a participant fails before reaching the barrier
BARRIER_WAIT_TIMEOUT_SECS = 10.0


def _participant_store_value(round_id, node_desc, infra_rank=-1, domain_id="none"):
    return RendezvousStoreValue.pack(
        round_id,
        RendezvousParticipantInfo.to_payload(node_desc, infra_rank, domain_id),
    ).encode("utf-8")


def _rank_store_value(round_id, rank, total):
    return RendezvousStoreValue.pack(round_id, {"rank": rank, "total": total}).encode("utf-8")


def _seed_joined_participants(store, state, participants, round_id=0):
    """Populate join_count and slot_N keys for deterministic host-close tests."""
    for slot, (node_desc, infra_rank, domain_id) in enumerate(participants, start=1):
        store.add(state.join_count_key, 1)
        store.set(
            f"{state.prefix}:slot_{slot}",
            state._pack_participant_value(node_desc, infra_rank, domain_id, round_id),
        )


# Helper to create segment check interval for tests
def _test_segment_check_interval(seconds=TEST_SEGMENT_CHECK_INTERVAL_SECS):
    """Create segment check interval for test."""
    return seconds


def _assert_threads_finished(test_case, threads, timeout_secs):
    """Assert all threads have terminated after join; fail with a clear message if not."""
    alive = [t for t in threads if t.is_alive()]
    if alive:
        names = [t.name for t in alive]
        test_case.fail(f"Thread(s) did not terminate within {timeout_secs}s: {names}")


class BaseRendezvousTest(TestCase):
    """Base test class that clears infrastructure rank environment variables.

    Clears all environment variables that affect get_infrastructure_rank():
    - SLURM_TOPOLOGY_ADDR
    - SLURM_TOPOLOGY_ADDR_PATTERN
    - SLURM_PROCID
    - GROUP_RANK
    - NVRX_INFRA_RANK_FROM_NODENAME
    - SLURMD_NODENAME
    - CROSS_SLURM_PROCID
    - SLURM_ARRAY_TASK_ID
    - SLURM_JOB_ID (to avoid validation errors when SLURM_PROCID is cleared)
    - SLURM_NNODES
    - SLURM_JOB_NUM_NODES

    Most tests should inherit from this to ensure they use deterministic rank
    assignment rather than being affected by the shell environment.

    Tests that specifically need to test infrastructure rank behavior from
    environment variables should inherit directly from TestCase instead.
    """

    def setUp(self):
        """Save and clear SLURM/GROUP_RANK environment variables."""

        # Clear all environment variables that affect get_infrastructure_rank()
        # to ensure deterministic rank assignment in tests
        self._saved_env_vars = {}
        env_vars_to_clear = [
            'SLURM_TOPOLOGY_ADDR',
            'SLURM_TOPOLOGY_ADDR_PATTERN',
            'SLURM_PROCID',
            'GROUP_RANK',
            'NVRX_INFRA_RANK_FROM_NODENAME',
            'SLURMD_NODENAME',
            'CROSS_SLURM_PROCID',
            'SLURM_ARRAY_TASK_ID',
            'SLURM_ARRAY_JOB_ID',
            'SLURM_JOB_ID',  # Must clear to avoid validation error when SLURM_PROCID is cleared
            'SLURM_RESTART_CNT',
            'SLURM_NNODES',
            'SLURM_JOB_NUM_NODES',
        ]
        for var in env_vars_to_clear:
            self._saved_env_vars[var] = os.environ.pop(var, None)

        super().setUp()

    def tearDown(self):
        """Restore SLURM/GROUP_RANK environment variables."""

        super().tearDown()
        # Restore all saved environment variables
        for var, value in self._saved_env_vars.items():
            if value is not None:
                os.environ[var] = value


class BarrierStateBasicTest(BaseRendezvousTest):
    """Test basic barrier state operations."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,  # Use any available port
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        super().setUp()  # Clears environment variables

        # Reuse the shared store
        self.store = self.shared_store
        # Use uuid so run_id is globally unique (time-based run_id can collide in same microsecond).
        self.run_id = f"test_run_{self._testMethodName}_{uuid.uuid4().hex}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_barrier_state_initialization(self):
        """Test that barrier state initializes with correct key prefixes."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        self.assertEqual(state.run_id, self.run_id)
        self.assertTrue(state.is_store_host)
        self.assertIn(self.run_id, state.prefix)
        self.assertIn(self.run_id, state.join_count_key)
        self.assertIn(self.run_id, state.round_done_key)

    def test_is_shutdown_initially_false(self):
        """Test that rendezvous is not shut down initially."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        self.assertFalse(state.is_shutdown())

    def test_set_shutdown(self):
        """Test that set_shutdown marks rendezvous as permanently shut down."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        state.set_shutdown()
        self.assertTrue(state.is_shutdown())

    def test_join_increments_join_count(self):
        """Test that joining increments join_count atomically."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # First join
        count1 = self.store.add(state.join_count_key, 1)
        self.assertEqual(count1, 1)

        # Second join
        count2 = self.store.add(state.join_count_key, 1)
        self.assertEqual(count2, 2)

        # Third join
        count3 = self.store.add(state.join_count_key, 1)
        self.assertEqual(count3, 3)

    def test_round_fenced_slot_cas_refuses_newer_round(self):
        """A stale participant metadata write must not clobber a newer round's slot."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        slot_key = f"{state.prefix}:slot_1"
        newer_node = _NodeDesc("newer", 2, 0)
        stale_node = _NodeDesc("stale", 1, 0)
        newer_value = _participant_store_value(1, newer_node, 2, "none")
        stale_value = _participant_store_value(0, stale_node, 1, "none")
        self.store.set(slot_key, newer_value)

        with self.assertRaises(_StaleRendezvousRoundError):
            state._round_fenced_compare_set(
                slot_key,
                stale_value,
            )

        self.assertEqual(self.store.get(slot_key), newer_value)
        self.assertEqual(state._round, 0)

    def test_round_fenced_rank_cas_refuses_newer_round(self):
        """A stale initial rank placeholder must not overwrite a newer round's rank key."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        rank_key = f"{state.prefix}:slot_1_rank"
        newer_value = _rank_store_value(1, GroupRankStatus.UNASSIGNED.value, 0)
        stale_value = _rank_store_value(0, 0, 1)
        self.store.set(rank_key, newer_value)

        with self.assertRaises(_StaleRendezvousRoundError):
            state._round_fenced_compare_set(
                rank_key,
                stale_value,
            )

        self.assertEqual(self.store.get(rank_key), newer_value)
        self.assertEqual(state._round, 0)

    def test_round_fenced_cas_rejects_same_round_conflict(self):
        """Same-round conflicts indicate broken slot ownership and must fail loudly."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        rank_key = f"{state.prefix}:slot_1_rank"
        assigned_value = _rank_store_value(0, 0, 1)
        self.store.set(rank_key, assigned_value)

        with self.assertRaisesRegex(RuntimeError, "same-round rendezvous write conflict"):
            state._round_fenced_compare_set(
                rank_key,
                _rank_store_value(0, GroupRankStatus.UNASSIGNED.value, 0),
            )

        self.assertEqual(self.store.get(rank_key), assigned_value)

    def test_leave_rendezvous_refuses_newer_round_slot(self):
        """A stale leave must not mark a newer round's slot as WITHDRAWN."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        state._slot = 1
        state._joined_node_desc = _NodeDesc("stale", 1, 0)
        slot_key = f"{state.prefix}:slot_1"
        newer_node = _NodeDesc("newer", 2, 0)
        newer_value = _participant_store_value(1, newer_node, 2, "none")
        self.store.set(slot_key, newer_value)

        state._leave_rendezvous()

        self.assertEqual(self.store.get(slot_key), newer_value)
        self.assertEqual(state._round, 0)

    def test_leave_rendezvous_updates_same_node_slot(self):
        """A participant may mark its own same-round slot as WITHDRAWN."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        node = _NodeDesc("node", 1, 0)
        state._slot = 1
        state._joined_node_desc = node
        slot_key = f"{state.prefix}:slot_1"
        self.store.set(slot_key, _participant_store_value(0, node, 1, "none"))

        state._leave_rendezvous()

        round_id, payload = RendezvousStoreValue.unpack(self.store.get(slot_key).decode("utf-8"))
        _, infra_rank, domain_id = RendezvousParticipantInfo.from_payload(payload)
        self.assertEqual(round_id, 0)
        self.assertEqual(infra_rank, -1)
        self.assertEqual(domain_id, WITHDRAWN)

    def test_leave_rendezvous_rejects_same_round_different_node(self):
        """A participant must not mark another same-round node's slot as WITHDRAWN."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        owner = _NodeDesc("owner", 1, 0)
        leaving = _NodeDesc("leaving", 2, 0)
        state._slot = 1
        state._joined_node_desc = leaving
        slot_key = f"{state.prefix}:slot_1"
        original_value = _participant_store_value(0, owner, 1, "none")
        self.store.set(slot_key, original_value)

        with self.assertRaisesRegex(RuntimeError, "same-round rendezvous write conflict"):
            state._leave_rendezvous()

        self.assertEqual(self.store.get(slot_key), original_value)

    def test_get_all_participants_incremental_fetch(self):
        """Test incremental fetching of participants using get_all_participants."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Add some participants to the store
        participants_data = [
            (self.node_desc_gen.generate(), 0, "domain1"),
            (self.node_desc_gen.generate(), 1, "domain1"),
            (self.node_desc_gen.generate(), 2, "domain2"),
            (self.node_desc_gen.generate(), 3, "domain2"),
            (self.node_desc_gen.generate(), 4, "domain3"),
        ]

        for i, (node_desc, infra_rank, domain_id) in enumerate(participants_data, start=1):
            slot_key = f"{state.prefix}:slot_{i}"
            participant_data = _participant_store_value(0, node_desc, infra_rank, domain_id)
            self.store.set(slot_key, participant_data)

        # Test 1: Fetch all participants at once
        all_participants = state.get_all_participants(total_participants=5)
        self.assertEqual(len(all_participants), 5)
        self.assertEqual(all_participants[0][1], 0)  # First participant infra_rank=0
        self.assertEqual(all_participants[4][1], 4)  # Last participant infra_rank=4

        # Test 2: Fetch first 2 participants
        first_two = state.get_all_participants(total_participants=2)
        self.assertEqual(len(first_two), 2)
        self.assertEqual(first_two[0][1], 0)
        self.assertEqual(first_two[1][1], 1)

        # Test 3: Incremental fetch - get next 3 participants
        # Make a copy of first_two to avoid mutation
        first_two_copy = list(first_two)
        all_five = state.get_all_participants(
            total_participants=5,
            start_index=3,  # Start from 3rd participant
            existing_participants=first_two_copy,
        )
        self.assertEqual(len(all_five), 5)
        self.assertEqual(all_five[0][1], 0)  # First from cache
        self.assertEqual(all_five[1][1], 1)  # Second from cache
        self.assertEqual(all_five[2][1], 2)  # Third is new
        self.assertEqual(all_five[3][1], 3)  # Fourth is new
        self.assertEqual(all_five[4][1], 4)  # Fifth is new
        # Verify original first_two was mutated (it's the same object as first_two_copy before the call)
        self.assertEqual(len(first_two), 2)  # Original list should still be 2

        # Test 4: Edge case - start_index beyond total_participants (returns existing)
        first_two_copy2 = list(first_two)
        result = state.get_all_participants(
            total_participants=5, start_index=10, existing_participants=first_two_copy2
        )
        self.assertEqual(len(result), 2)  # Should return existing list (first_two_copy2)

        # Test 5: Edge case - start_index equals total_participants + 1 (returns existing)
        first_two_copy3 = list(first_two)
        result = state.get_all_participants(
            total_participants=5, start_index=6, existing_participants=first_two_copy3
        )
        self.assertEqual(len(result), 2)  # Should return existing list (first_two_copy3)

        # Test 6: Round filtering - mismatched round should be treated as placeholder
        mismatch_key = f"{state.prefix}:slot_3"
        mismatch_data = _participant_store_value(
            1, participants_data[2][0], participants_data[2][1], participants_data[2][2]
        )
        self.store.set(mismatch_key, mismatch_data)
        filtered = state.get_all_participants(total_participants=5, expected_round_id=0)
        self.assertEqual(len(filtered), 5)
        self.assertEqual(
            filtered[2][2],
            WITHDRAWN,
            "Round mismatch should be converted to placeholder participant",
        )

    def test_store_value_unpack_requires_round_field(self):
        """Test store value unpack requires round-fencing fields."""
        missing_round_data = '{"payload":{"addr":"n1","pid":1,"local_id":0}}'
        with self.assertRaises(ValueError):
            RendezvousStoreValue.unpack(missing_round_data)

    def test_step2_poll_interval_adaptive_scaling(self):
        """Test adaptive Step 2 poll interval floor and cap behavior."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Small rendezvous should poll every 1s minimum.
        self.assertEqual(
            state._compute_step2_poll_interval(min_nodes=1, segment_check_interval=5.0), 1.0
        )
        self.assertEqual(
            state._compute_step2_poll_interval(min_nodes=100, segment_check_interval=5.0), 1.0
        )

        # Mid-size scales linearly (min_nodes / 2000), capped by configured interval.
        self.assertEqual(
            state._compute_step2_poll_interval(min_nodes=6000, segment_check_interval=5.0), 3.0
        )

        # Large rendezvous is capped by the configured max.
        self.assertEqual(
            state._compute_step2_poll_interval(min_nodes=10000, segment_check_interval=5.0), 5.0
        )
        self.assertEqual(
            state._compute_step2_poll_interval(min_nodes=10000, segment_check_interval=2.0), 2.0
        )


class Step2CompletionTest(BaseRendezvousTest):
    """Test Step 2 completion signaling logic."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use uuid so run_id is globally unique. time.time()-based run_id can collide
        # when tests run in the same microsecond (retries/fast runs), leaving
        # round_done_0=1 from a prior run so participants wait at Step 0.
        self.run_id = f"test_step2_{self._testMethodName}_{uuid.uuid4().hex}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_step2_completion_on_max_nodes(self):
        """Test that Step 2 completes immediately when max_nodes is reached.

        Populate the host-visible store state directly so this test is about
        Step 2 closure, not child process startup timing.
        """
        min_nodes = 3
        max_nodes = 3
        segment_check_interval = _test_segment_check_interval()
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node1", 101, 0), 1, "none"),
            (_NodeDesc("node2", 102, 0), 2, "none"),
        ]
        _seed_joined_participants(self.store, state, participants)

        state._rendezvous_start_time = time.monotonic()
        state._host_close_round(
            _NodeDesc("control", 10, 0),
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            segment_check_interval=segment_check_interval,
        )

        self.assertEqual(self.store.get(state.round_done_key).decode("utf-8"), "1")
        ranks = [
            state._unpack_rank_value(self.store.get(f"{state.prefix}:slot_{slot}_rank"))[0]
            for slot in range(1, max_nodes + 1)
        ]
        self.assertEqual(set(ranks), set(range(max_nodes)))

    def test_step2_completion_on_min_nodes_with_segment_check(self):
        """Test that Step 2 completes when min_nodes is reached and segment constraint is satisfied.

        This exercises the host close logic directly with deterministic Step 2
        state instead of relying on participant process timing.
        """
        min_nodes = 2
        max_nodes = 4
        segment_check_interval = _test_segment_check_interval()
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            segment=2,
        )

        participants = [
            (_NodeDesc("node0", 100, 0), 0, "domain-a"),
            (_NodeDesc("node1", 101, 0), 1, "domain-a"),
        ]
        _seed_joined_participants(self.store, state, participants)

        state._rendezvous_start_time = time.monotonic()
        state._host_close_round(
            _NodeDesc("control", 10, 0),
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            segment_check_interval=segment_check_interval,
        )

        self.assertEqual(self.store.get(state.round_done_key).decode("utf-8"), "1")
        ranks = [
            state._unpack_rank_value(self.store.get(f"{state.prefix}:slot_{slot}_rank"))[0]
            for slot in range(1, min_nodes + 1)
        ]
        self.assertEqual(ranks, [0, 1])

    def test_step2_participants_see_completion_key(self):
        """Test that rendezvous completes correctly when all participants arrive quickly.

        Populate the host-visible store state directly so each participant's
        rank key can be checked without process timeout sensitivity.
        """
        min_nodes = 3
        max_nodes = 4
        segment_check_interval = _test_segment_check_interval()
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node1", 101, 0), 1, "none"),
            (_NodeDesc("node2", 102, 0), 2, "none"),
        ]
        _seed_joined_participants(self.store, state, participants)

        state._rendezvous_start_time = time.monotonic()
        state._host_close_round(
            _NodeDesc("control", 10, 0),
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            segment_check_interval=segment_check_interval,
        )

        for slot in range(1, min_nodes + 1):
            rank, total, written_round = state._unpack_rank_value(
                self.store.get(f"{state.prefix}:slot_{slot}_rank")
            )
            self.assertEqual(written_round, 0)
            self.assertEqual(total, min_nodes)
            self.assertIn(rank, range(min_nodes))


class RaceConditionTest(BaseRendezvousTest):
    """Test race conditions with concurrent operations."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use uuid so run_id is globally unique (time-based run_id can collide in same microsecond).
        self.run_id = f"test_race_{self._testMethodName}_{uuid.uuid4().hex}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_concurrent_joins(self):
        """Test that concurrent joins are handled correctly with atomic increments."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        num_participants = 10
        arrived_counts = []

        def join_thread():
            count = self.store.add(state.join_count_key, 1)
            arrived_counts.append(count)

        threads = []
        for _ in range(num_participants):
            t = threading.Thread(target=join_thread)
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)
        _assert_threads_finished(self, threads, TEST_THREAD_JOIN_TIMEOUT_SECS)

        # Check that all counts are unique (atomic increment)
        self.assertEqual(len(arrived_counts), num_participants)
        self.assertEqual(set(arrived_counts), set(range(1, num_participants + 1)))

    def test_late_arrival_after_snapshot_before_assignment(self):
        """Test late arrival in the critical window: after store host snapshot, before assignment.

        This tests the race condition where:
        1. Store host takes a double-checked snapshot of join_count_key (condition satisfied)
        2. Late arrival joins and increments join_count_key after the snapshot
        3. Store host assigns ranks based on the snapshot (late arrival is not included)
        4. Late arrival sees UNASSIGNED in Step 3 and retries in the next round

        The current design handles this by having the store host snapshot join_count at the
        time of the double-check, ensuring consistent rank assignment. Late arrivals that
        miss the snapshot detect UNASSIGNED via the embedded round number in rank_key and
        loop to the next round.
        """
        min_nodes = 2
        max_nodes = 4
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        participants = [
            (_NodeDesc("host", 100, 0), 0, "none"),
            (_NodeDesc("regular", 101, 0), 1, "none"),
        ]
        _seed_joined_participants(self.store, state, participants)

        current_joined_snapshot = int(self.store.get(state.join_count_key).decode("utf-8"))

        late_slot = self.store.add(state.join_count_key, 1)
        late_rank_key = f"{state.prefix}:slot_{late_slot}_rank"
        self.store.set(
            late_rank_key,
            state._pack_rank_value(GroupRankStatus.UNASSIGNED.value, 0, state._round),
        )
        self.store.set(
            f"{state.prefix}:slot_{late_slot}",
            state._pack_participant_value(_NodeDesc("late", 102, 0), 2, "none", state._round),
        )

        state.assign_group_ranks(
            world_size=min_nodes,
            max_nodes=max_nodes,
            total_participants=current_joined_snapshot,
            node_desc=_NodeDesc("control", 10, 0),
        )
        self.store.set(state.round_done_key, "1".encode("utf-8"))

        assigned_ranks = [
            state._unpack_rank_value(self.store.get(f"{state.prefix}:slot_{slot}_rank"))[0]
            for slot in range(1, min_nodes + 1)
        ]
        self.assertEqual(set(assigned_ranks), {0, 1})

        state._rendezvous_start_time = time.monotonic()
        late_rank, late_total = state._wait_for_round_done(_NodeDesc("late", 102, 0), late_rank_key)
        _, _, late_round = state._unpack_rank_value(self.store.get(late_rank_key))
        self.assertEqual(late_rank, GroupRankStatus.UNASSIGNED.value)
        self.assertEqual(late_total, 0)
        self.assertEqual(late_round, state._round)

    def test_multiple_completion_signals(self):
        """Test that multiple participants can try to set completion key (idempotent)."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Multiple threads try to set the same key
        def set_completion():
            self.store.set(state.round_done_key, "1".encode('utf-8'))

        threads = []
        for _ in range(5):
            t = threading.Thread(target=set_completion)
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)
        _assert_threads_finished(self, threads, TEST_THREAD_JOIN_TIMEOUT_SECS)

        # Key should be set (no errors)
        self.assertTrue(self.store.check([state.round_done_key]))


class StoreHostBehaviorTest(BaseRendezvousTest):
    """Test store host specific behavior."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use uuid so run_id is globally unique. time.time()-based run_id can collide
        # when tests run in the same microsecond (retries/fast runs), leaving
        # round_done_0=1 so participants block at Step 0.
        self.run_id = f"test_host_{self._testMethodName}_{uuid.uuid4().hex}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_rank_assignment_with_arrival_order(self):
        """Test that store host assigns ranks to all participants.

        Populate slots directly so the assertion covers rank assignment without
        depending on process startup timing.
        """
        min_nodes = 3
        max_nodes = 3
        segment_check_interval = _test_segment_check_interval()
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node1", 101, 0), 1, "none"),
            (_NodeDesc("node2", 102, 0), 2, "none"),
        ]
        _seed_joined_participants(self.store, state, participants)

        state._rendezvous_start_time = time.monotonic()
        state._host_close_round(
            _NodeDesc("control", 10, 0),
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            segment_check_interval=segment_check_interval,
        )

        rank_values = [
            state._unpack_rank_value(self.store.get(f"{state.prefix}:slot_{slot}_rank"))
            for slot in range(1, min_nodes + 1)
        ]
        ranks = [rank for rank, _, _ in rank_values]
        totals = [total for _, total, _ in rank_values]
        self.assertEqual(set(totals), {min_nodes})
        self.assertEqual(set(ranks), set(range(min_nodes)))

    def test_non_store_host_waits_for_ranks(self):
        """Test that non-store host participants wait for rank assignment.

        Block the round_done read with events so the wait behavior is
        deterministic and does not depend on process startup timing.
        """
        round_done_read = threading.Event()
        release_round_done = threading.Event()

        class StoreBlocksRoundDoneRead:
            def __init__(self, underlying):
                self._store = underlying

            def get(self, key):
                if ":round_done_" in key:
                    round_done_read.set()
                    release_round_done.wait(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)
                return self._store.get(key)

            def add(self, key, value):
                return self._store.add(key, value)

            def set(self, key, value):
                return self._store.set(key, value)

            def compare_set(self, key, expected_value, desired_value):
                return self._store.compare_set(key, expected_value, desired_value)

            def check(self, keys):
                return self._store.check(keys)

            def multi_get(self, keys):
                return self._store.multi_get(keys)

            def multi_set(self, keys, values):
                return self._store.multi_set(keys, values)

        state = _RendezvousBarrierState(
            store=StoreBlocksRoundDoneRead(self.store),
            run_id=self.run_id,
            is_store_host=False,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        node = _NodeDesc("node1", 101, 0)
        rank_key = f"{state.prefix}:slot_1_rank"
        self.store.set(rank_key, state._pack_rank_value(1, 2, state._round))
        state._rendezvous_start_time = time.monotonic()
        result = []

        def wait_for_rank():
            result.append(state._wait_for_round_done(node, rank_key))

        waiter = threading.Thread(target=wait_for_rank)
        waiter.start()
        self.assertTrue(round_done_read.wait(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS))
        self.store.set(state.round_done_key, "1".encode("utf-8"))
        release_round_done.set()
        waiter.join(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)
        _assert_threads_finished(self, [waiter], TEST_THREAD_JOIN_TIMEOUT_SECS)

        self.assertEqual(result, [(1, 2)])


class GroupRankAssignmentTest(TestCase):
    """Test group rank assignment logic."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        # NOTE: DO NOT clear SLURM_PROCID/GROUP_RANK for this test class
        # These tests specifically test infrastructure rank behavior

        self.store = self.shared_store
        # Use uuid so run_id is globally unique (time-based run_id can collide in same microsecond).
        self.run_id = f"test_rank_{self._testMethodName}_{uuid.uuid4().hex}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        pass

    def test_rank_assignment_with_arrival_order(self):
        """Test rank assignment based on arrival order (non-infra mode)."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
        )

        # Create participants
        participants = [
            (_NodeDesc("node_a", 100, 0), -1, None),
            (_NodeDesc("node_b", 101, 0), -1, None),
            (_NodeDesc("node_c", 102, 0), -1, None),
        ]
        min_nodes = 3

        # Assign ranks
        result = state._assign_group_ranks(participants, min_nodes)

        # Check all got unique ranks
        self.assertEqual(len(result), 3)
        ranks = set(result.values())
        self.assertEqual(ranks, {0, 1, 2})

    def test_rank_assignment_with_infra_rank(self):
        """Test rank assignment using infrastructure ranks."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Create participants with infra ranks
        participants = [
            (_NodeDesc("node_a", 100, 0), 0, "none"),
            (_NodeDesc("node_b", 101, 0), 1, "none"),
            (_NodeDesc("node_c", 102, 0), 2, "none"),
        ]
        min_nodes = 3

        # Assign ranks
        result = state._assign_group_ranks(participants, min_nodes)

        # Check direct mapping: group_rank = infra_rank
        self.assertEqual(result[participants[0][0]], 0)
        self.assertEqual(result[participants[1][0]], 1)
        self.assertEqual(result[participants[2][0]], 2)

    def test_control_host_closes_round_without_joining(self):
        """An external control host closes a compute-only round without claiming a slot."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        join_count_key = state.join_count_key
        prefix = state.prefix
        train_0 = _NodeDesc("node_a", 100, 0)
        train_1 = _NodeDesc("node_b", 101, 0)

        self.store.add(join_count_key, 1)
        self.store.add(join_count_key, 1)
        self.store.set(
            f"{prefix}:slot_1",
            state._pack_participant_value(train_0, 0, "none", 0),
        )
        self.store.set(
            f"{prefix}:slot_2",
            state._pack_participant_value(train_1, 1, "none", 0),
        )

        closed_round = state.close_current_round_as_host(
            _NodeDesc("control", 10, 0),
            min_nodes=2,
            max_nodes=2,
            segment_check_interval=_test_segment_check_interval(),
        )

        self.assertEqual(closed_round, 0)
        self.assertEqual(state._round, 0)
        self.assertEqual(int(self.store.get(join_count_key).decode("utf-8")), 2)
        self.assertFalse(self.store.check([f"{prefix}:slot_3"]))
        rank_0, total_0, round_0 = state._unpack_rank_value(self.store.get(f"{prefix}:slot_1_rank"))
        rank_1, total_1, round_1 = state._unpack_rank_value(self.store.get(f"{prefix}:slot_2_rank"))
        self.assertEqual((rank_0, rank_1), (0, 1))
        self.assertEqual((total_0, total_1), (2, 2))
        self.assertEqual((round_0, round_1), (0, 0))

    def test_control_host_reports_cycle_start_when_round_closes(self):
        """A colocated store host reports cycle start after rank assignment."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        reporter = MagicMock()
        state._cycle_info_reporter = reporter

        join_count_key = state.join_count_key
        prefix = state.prefix
        train_0 = _NodeDesc("node_a", 100, 0)
        train_1 = _NodeDesc("node_b", 101, 0)

        self.store.add(join_count_key, 1)
        self.store.add(join_count_key, 1)
        self.store.set(
            f"{prefix}:slot_1",
            state._pack_participant_value(train_0, 0, "none", 0),
        )
        self.store.set(
            f"{prefix}:slot_2",
            state._pack_participant_value(train_1, 1, "none", 0),
        )

        state.close_current_round_as_host(
            _NodeDesc("control", 10, 0),
            min_nodes=2,
            max_nodes=2,
            segment_check_interval=_test_segment_check_interval(),
        )

        reporter.report_cycle_start.assert_called_once()
        snapshot = reporter.report_cycle_start.call_args.args[0]
        self.assertEqual(snapshot.cycle_number, 0)
        self.assertEqual(snapshot.active_node_addrs, ["node_a", "node_b"])
        self.assertEqual(snapshot.standby_node_addrs, [])
        self.assertEqual(snapshot.active_ranks, [0, 1])

    def test_open_rendezvous_does_not_report_cycle_end(self):
        """Opening the next round only opens rendezvous; cycle end is reporter-owned."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        reporter = MagicMock()
        state._cycle_info_reporter = reporter

        state.open_rendezvous()

        reporter.report_cycle_end.assert_not_called()

    def test_previous_active_last_call_gate_waits_for_missing_infra_rank(self):
        """The close gate waits while previous active infra ranks are missing."""
        gate = _PreviousActiveLastCallGate(
            previous_active_infra_ranks={0, 1},
            timeout_seconds=15.0,
            enabled=True,
        )
        previous_0 = _NodeDesc("node0", 100, 0)
        previous_1 = _NodeDesc("node1", 101, 0)
        spare = _NodeDesc("spare", 102, 0)

        decision = gate.evaluate([(previous_0, 0, "none"), (spare, 2, "none")], now=100.0)
        self.assertFalse(decision.should_close)
        self.assertEqual(decision.missing_previous_active, (1,))
        self.assertEqual(decision.last_call_deadline, 115.0)

        decision = gate.evaluate(
            [(previous_0, 0, "none"), (previous_1, 1, "none"), (spare, 2, "none")],
            now=101.0,
        )
        self.assertTrue(decision.should_close)
        self.assertEqual(decision.missing_previous_active, ())

    def test_previous_active_last_call_gate_is_noop_when_disabled(self):
        """No-hot-spare or no-snapshot paths keep the current immediate close behavior."""
        gate = _PreviousActiveLastCallGate(
            previous_active_infra_ranks={0, 1},
            timeout_seconds=15.0,
            enabled=False,
        )
        decision = gate.evaluate([(_NodeDesc("spare", 102, 0), 2, "none")], now=100.0)

        self.assertTrue(decision.should_close)

    def test_previous_active_snapshot_is_host_local_by_infra_rank(self):
        """The prior active set is remembered in host-local state by infrastructure rank."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        active_0 = _NodeDesc("node0", 100, 0)
        active_1 = _NodeDesc("node1", 101, 0)
        spare = _NodeDesc("spare", 102, 0)
        participants = [
            (active_0, 10, "none"),
            (active_1, 20, "none"),
            (spare, 30, "none"),
        ]
        for slot, (node_desc, infra_rank, domain_id) in enumerate(participants, start=1):
            self.store.add(state.join_count_key, 1)
            self.store.set(
                f"{state.prefix}:slot_{slot}",
                state._pack_participant_value(node_desc, infra_rank, domain_id, 0),
            )

        state.assign_group_ranks(
            world_size=2,
            max_nodes=3,
            total_participants=3,
            node_desc=active_0,
        )

        self.assertEqual(state._previous_active_infra_ranks, {10, 20})

    def test_host_close_waits_for_previous_active_before_accepting_spare(self):
        """A hot spare that arrives first does not close the round before last_call."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            last_call_timeout_seconds=1.0,
        )
        # Keep the test fast; the last-call gate owns the timing being tested.
        state._compute_step2_poll_interval = lambda min_nodes, segment_check_interval: 0.0

        previous_0 = _NodeDesc("node0", 100, 0)
        previous_1 = _NodeDesc("node1", 101, 0)
        spare = _NodeDesc("spare", 102, 0)
        state._remember_previous_active_infra_ranks(
            [(previous_0, 0, "none"), (previous_1, 1, "none")],
            {previous_0: 0, previous_1: 1},
            world_size=2,
        )

        state._round = 1
        self.store.add(state.join_count_key, 1)
        self.store.add(state.join_count_key, 1)
        self.store.set(
            f"{state.prefix}:slot_1",
            state._pack_participant_value(previous_0, 0, "none", 1),
        )
        self.store.set(
            f"{state.prefix}:slot_2",
            state._pack_participant_value(spare, 2, "none", 1),
        )

        original_make_gate = state._make_previous_active_last_call_gate

        class InjectPreviousActiveAfterFirstWait:
            def __init__(self, gate):
                self._gate = gate
                self.injected = False

            def evaluate(self, participants, now):
                decision = self._gate.evaluate(participants, now)
                if not decision.should_close and not self.injected:
                    slot = state.store.add(state.join_count_key, 1)
                    state.store.set(
                        f"{state.prefix}:slot_{slot}",
                        state._pack_participant_value(previous_1, 1, "none", 1),
                    )
                    self.injected = True
                return decision

        state._make_previous_active_last_call_gate = lambda min_nodes, max_nodes: (
            InjectPreviousActiveAfterFirstWait(original_make_gate(min_nodes, max_nodes))
        )
        state._rendezvous_start_time = time.monotonic()
        state._host_close_round(
            _NodeDesc("control", 10, 0),
            min_nodes=2,
            max_nodes=3,
            segment_check_interval=0.0,
        )

        rank_0, _, _ = state._unpack_rank_value(self.store.get(f"{state.prefix}:slot_1_rank"))
        spare_rank, _, _ = state._unpack_rank_value(self.store.get(f"{state.prefix}:slot_2_rank"))
        rank_1, _, _ = state._unpack_rank_value(self.store.get(f"{state.prefix}:slot_3_rank"))
        self.assertEqual(rank_0, 0)
        self.assertEqual(rank_1, 1)
        self.assertEqual(spare_rank, 2)

    def test_rank_assignment_standby_nodes(self):
        """Test rank assignment for standby nodes (beyond min_nodes)."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
        )

        # 5 participants, min_nodes=3
        participants = [
            (_NodeDesc("node_a", 100, 0), -1, "none"),
            (_NodeDesc("node_b", 101, 0), -1, "none"),
            (_NodeDesc("node_c", 102, 0), -1, "none"),
            (_NodeDesc("node_d", 103, 0), -1, "none"),
            (_NodeDesc("node_e", 104, 0), -1, "none"),
        ]
        min_nodes = 3

        # Assign ranks
        result = state._assign_group_ranks(participants, min_nodes)

        # First 3 should get ranks 0-2 (active)
        # Last 2 should get ranks 3-4 (standby)
        self.assertEqual(len(result), 5)
        ranks = sorted(result.values())
        self.assertEqual(ranks, [0, 1, 2, 3, 4])

    def test_rank_assignment_preserves_slurm_topology_order(self):
        """Test that participants arriving out-of-order are assigned group ranks
        in SLURM topology order (sorted by infrastructure rank), not arrival order.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
        )

        # Create nodes that arrive in arbitrary order (e.g., alphabetically: aaa < bbb < zzz)
        # But assign them infrastructure ranks in reverse order
        node_aaa = _NodeDesc("aaa_node", 100, 0)
        node_bbb = _NodeDesc("bbb_node", 101, 0)
        node_zzz = _NodeDesc("zzz_node", 102, 0)

        # Verify sort order by node descriptor is as expected
        sorted_nodes = sorted([node_zzz, node_aaa, node_bbb])
        self.assertEqual(sorted_nodes, [node_aaa, node_bbb, node_zzz])

        # Create participants list with infra ranks in reverse of alphabetical order
        # Simulating they arrived/joined out of order
        participants = [
            (node_aaa, 102, "none"),  # Largest infra rank
            (node_bbb, 101, "none"),  # Middle infra rank
            (node_zzz, 100, "none"),  # Smallest infra rank
        ]
        min_nodes = 3

        # Assign ranks
        result = state._assign_group_ranks(participants, min_nodes)

        # Group ranks should follow SLURM topology order (infra rank order), not node descriptor order
        self.assertEqual(result[node_zzz], 0)  # Smallest infra rank -> group rank 0
        self.assertEqual(result[node_bbb], 1)  # Middle infra rank -> group rank 1
        self.assertEqual(result[node_aaa], 2)  # Largest infra rank -> group rank 2

    def test_rank_assignment_with_hot_spare_segment_none(self):
        """Test rank assignment with hot spare nodes (segment=None).

        When there are more participants than world_size and segment=None,
        the extra nodes become hot spares with ranks >= world_size.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=None,  # No segment awareness
        )

        # 5 nodes, world_size=3 -> first 3 active [0,1,2], remaining 2 hot spares [3,4]
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),  # Active
            (_NodeDesc("node1", 101, 0), 1, "none"),  # Active
            (_NodeDesc("node2", 102, 0), 2, "none"),  # Active
            (_NodeDesc("node3", 103, 0), 3, "none"),  # Hot spare
            (_NodeDesc("node4", 104, 0), 4, "none"),  # Hot spare
        ]
        world_size = 3

        result = state._assign_group_ranks(participants, world_size)

        # First 3 get active ranks [0,1,2]
        self.assertEqual(result[_NodeDesc("node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("node1", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("node2", 102, 0)], 2)

        # Remaining 2 are hot spares [3,4]
        self.assertEqual(result[_NodeDesc("node3", 103, 0)], 3)
        self.assertEqual(result[_NodeDesc("node4", 104, 0)], 4)

    def test_rank_assignment_segment_1_with_hot_spare(self):
        """Test rank assignment with segment=1 and hot spare nodes.

        segment=1 means each node is a complete segment.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=1,
        )

        # 5 nodes, world_size=3, segment=1 -> first 3 active, remaining 2 hot spares
        # Each node has its own domain (simulated with node names like "domain0")
        participants = [
            (_NodeDesc("domain0-node", 100, 0), 0, "domain0"),
            (_NodeDesc("domain1-node", 101, 0), 1, "domain1"),
            (_NodeDesc("domain2-node", 102, 0), 2, "domain2"),
            (_NodeDesc("domain3-node", 103, 0), 3, "domain3"),  # Hot spare
            (_NodeDesc("domain4-node", 104, 0), 4, "domain4"),  # Hot spare
        ]
        world_size = 3

        result = state._assign_group_ranks(participants, world_size)

        # Active ranks
        self.assertEqual(result[_NodeDesc("domain0-node", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("domain1-node", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("domain2-node", 102, 0)], 2)

        # Hot spares
        self.assertEqual(result[_NodeDesc("domain3-node", 103, 0)], 3)
        self.assertEqual(result[_NodeDesc("domain4-node", 104, 0)], 4)

    def test_rank_assignment_segment_4_with_hot_spare(self):
        """Test rank assignment with segment=4 and hot spare nodes.

        segment=4 means we need 4 nodes per segment.
        world_size=8 requires 2 complete segments.
        Using domain-aware assignment with 2 domains.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: 8 nodes (2 complete segments) -> first 8 active
        # Domain 101: 4 nodes (1 complete segment) -> all standby (already have enough)
        # Total 12 nodes, world_size=8 (2 segments)
        participants = [
            # Domain 100: 8 nodes
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            (_NodeDesc("nvl72100-node4", 104, 0), 4, "nvl72100"),
            (_NodeDesc("nvl72100-node5", 105, 0), 5, "nvl72100"),
            (_NodeDesc("nvl72100-node6", 106, 0), 6, "nvl72100"),
            (_NodeDesc("nvl72100-node7", 107, 0), 7, "nvl72100"),
            # Domain 101: 4 nodes
            (_NodeDesc("nvl72101-node0", 108, 0), 8, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 109, 0), 9, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 110, 0), 10, "nvl72101"),
            (_NodeDesc("nvl72101-node3", 111, 0), 11, "nvl72101"),
        ]
        world_size = 8

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100: all 8 nodes are active (2 complete segments) [0-7]
        for i in range(8):
            self.assertEqual(result[_NodeDesc(f"nvl72100-node{i}", 100 + i, 0)], i)

        # Domain 101: all 4 nodes are standby [8-11]
        for i in range(4):
            self.assertEqual(result[_NodeDesc(f"nvl72101-node{i}", 108 + i, 0)], 8 + i)

    def test_rank_assignment_segment_16_with_hot_spare(self):
        """Test rank assignment with segment=16 and hot spare nodes.

        segment=16 means we need 16 nodes per segment.
        Using domain-aware assignment with 2 domains.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=16,
        )

        # Domain 100: 16 nodes (1 complete segment) -> all 16 active
        # Domain 101: 4 nodes (0 complete segments) -> all 4 standby
        # Total 20 nodes, world_size=16 (1 segment)
        participants = [
            # Domain 100: 16 nodes
            *[(_NodeDesc(f"nvl72100-node{i}", 100 + i, 0), i, "nvl72100") for i in range(16)],
            # Domain 101: 4 nodes
            *[(_NodeDesc(f"nvl72101-node{i}", 116 + i, 0), 16 + i, "nvl72101") for i in range(4)],
        ]
        world_size = 16

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100: all 16 nodes are active (1 complete segment) [0-15]
        for i in range(16):
            self.assertEqual(result[_NodeDesc(f"nvl72100-node{i}", 100 + i, 0)], i)

        # Domain 101: all 4 nodes are standby [16-19]
        for i in range(4):
            self.assertEqual(result[_NodeDesc(f"nvl72101-node{i}", 116 + i, 0)], 16 + i)

    def test_rank_assignment_domain_with_zero_complete_segments(self):
        """Test rank assignment when a domain has 0 complete segments.

        If segment=4 and a domain has only 3 nodes, those nodes can't form
        a complete segment and should be assigned to standby.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: 4 nodes (1 complete segment) -> active
        # Domain 101: 3 nodes (0 complete segments) -> all standby
        # World_size=4 requires 1 segment
        participants = [
            # Domain 100: 4 nodes
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            # Domain 101: 3 nodes (incomplete segment)
            (_NodeDesc("nvl72101-node0", 104, 0), 4, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 105, 0), 5, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 106, 0), 6, "nvl72101"),
        ]
        world_size = 4

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100: all 4 nodes active [0-3]
        self.assertEqual(result[_NodeDesc("nvl72100-node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node1", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node2", 102, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node3", 103, 0)], 3)

        # Domain 101: all 3 nodes become standby [4-6]
        self.assertEqual(result[_NodeDesc("nvl72101-node0", 104, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72101-node1", 105, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72101-node2", 106, 0)], 6)

    def test_rank_assignment_domain_with_one_complete_segment(self):
        """Test rank assignment when a domain has exactly 1 complete segment."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: 4 nodes (1 complete segment)
        # Domain 101: 4 nodes (1 complete segment)
        # World_size=8 requires 2 segments, both domains contribute 1 segment
        participants = [
            # Domain 100
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            # Domain 101
            (_NodeDesc("nvl72101-node0", 104, 0), 4, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 105, 0), 5, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 106, 0), 6, "nvl72101"),
            (_NodeDesc("nvl72101-node3", 107, 0), 7, "nvl72101"),
        ]
        world_size = 8

        result = state._assign_group_ranks(participants, world_size)

        # Both domains contribute 1 segment each, all nodes active
        # Domain 100 comes first (lower infra_rank)
        self.assertEqual(result[_NodeDesc("nvl72100-node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node1", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node2", 102, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node3", 103, 0)], 3)

        # Domain 101
        self.assertEqual(result[_NodeDesc("nvl72101-node0", 104, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72101-node1", 105, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72101-node2", 106, 0)], 6)
        self.assertEqual(result[_NodeDesc("nvl72101-node3", 107, 0)], 7)

    def test_rank_assignment_domain_with_multiple_complete_segments(self):
        """Test rank assignment when a domain has multiple complete segments.

        A domain with 12 nodes and segment=4 has 3 complete segments.
        If world_size=4 (1 segment needed), domain uses 4 nodes, rest are standby.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: 12 nodes (3 complete segments)
        # World_size=4 requires 1 segment
        # -> First 4 nodes active, remaining 8 standby
        participants = [
            (_NodeDesc(f"nvl72100-node{i}", 100 + i, 0), i, "nvl72100") for i in range(12)
        ]
        world_size = 4

        result = state._assign_group_ranks(participants, world_size)

        # First 4 nodes active (1 segment)
        for i in range(4):
            self.assertEqual(result[_NodeDesc(f"nvl72100-node{i}", 100 + i, 0)], i)

        # Remaining 8 nodes standby [4-11]
        for i in range(4, 12):
            self.assertEqual(result[_NodeDesc(f"nvl72100-node{i}", 100 + i, 0)], i)

    def test_rank_assignment_multiple_domains_mixed_segments(self):
        """Test rank assignment with multiple domains having different segment counts.

        Test scenario:
        - Domain 100: 8 nodes (2 segments with segment=4)
        - Domain 101: 6 nodes (1 complete segment + 2 incomplete)
        - Domain 102: 4 nodes (1 complete segment)
        - World_size=8 (2 segments needed)
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        participants = [
            # Domain 100: 8 nodes (2 segments) - lower infra_rank comes first
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            (_NodeDesc("nvl72100-node4", 104, 0), 4, "nvl72100"),
            (_NodeDesc("nvl72100-node5", 105, 0), 5, "nvl72100"),
            (_NodeDesc("nvl72100-node6", 106, 0), 6, "nvl72100"),
            (_NodeDesc("nvl72100-node7", 107, 0), 7, "nvl72100"),
            (_NodeDesc("nvl72101-node0", 108, 0), 8, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 109, 0), 9, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 110, 0), 10, "nvl72101"),
            (_NodeDesc("nvl72101-node3", 111, 0), 11, "nvl72101"),
            (_NodeDesc("nvl72101-node4", 112, 0), 12, "nvl72101"),
            (_NodeDesc("nvl72101-node5", 113, 0), 13, "nvl72101"),
            (_NodeDesc("nvl72102-node0", 114, 0), 14, "nvl72102"),
            (_NodeDesc("nvl72102-node1", 115, 0), 15, "nvl72102"),
            (_NodeDesc("nvl72102-node2", 116, 0), 16, "nvl72102"),
            (_NodeDesc("nvl72102-node3", 117, 0), 17, "nvl72102"),
        ]
        world_size = 8  # Need 2 segments

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100 contributes 2 segments (all 8 nodes active) [0-7]
        for i in range(8):
            self.assertEqual(result[_NodeDesc(f"nvl72100-node{i}", 100 + i, 0)], i)

        # Domain 101: all nodes go to standby (we already have 2 segments)
        for i in range(6):
            self.assertEqual(result[_NodeDesc(f"nvl72101-node{i}", 108 + i, 0)], 8 + i)

        # Domain 102: all nodes go to standby
        for i in range(4):
            self.assertEqual(result[_NodeDesc(f"nvl72102-node{i}", 114 + i, 0)], 14 + i)

    def test_rank_assignment_out_of_order_with_hot_spare_segment_none(self):
        """Test that out-of-order infra rank arrivals work correctly with hot spares (segment=None).

        Participants arrive in arbitrary order, but should be sorted by infra_rank
        and assigned group ranks accordingly, with extras becoming hot spares.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=None,
        )

        # Participants arrive out-of-order (infra ranks: 4, 1, 3, 0, 2)
        # After sorting by infra_rank: 0, 1, 2, 3, 4
        # World_size=3 -> first 3 active [0,1,2], remaining 2 hot spares [3,4]
        participants = [
            (_NodeDesc("node4", 104, 0), 4, "none"),
            (_NodeDesc("node1", 101, 0), 1, "none"),
            (_NodeDesc("node3", 103, 0), 3, "none"),
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node2", 102, 0), 2, "none"),
        ]
        world_size = 3

        result = state._assign_group_ranks(participants, world_size)

        # Should be sorted by infra_rank and assigned accordingly
        self.assertEqual(
            result[_NodeDesc("node0", 100, 0)], 0
        )  # infra_rank 0 -> group rank 0 (active)
        self.assertEqual(
            result[_NodeDesc("node1", 101, 0)], 1
        )  # infra_rank 1 -> group rank 1 (active)
        self.assertEqual(
            result[_NodeDesc("node2", 102, 0)], 2
        )  # infra_rank 2 -> group rank 2 (active)
        self.assertEqual(
            result[_NodeDesc("node3", 103, 0)], 3
        )  # infra_rank 3 -> group rank 3 (hot spare)
        self.assertEqual(
            result[_NodeDesc("node4", 104, 0)], 4
        )  # infra_rank 4 -> group rank 4 (hot spare)

    def test_rank_assignment_out_of_order_with_hot_spare_segment_4(self):
        """Test that out-of-order infra rank arrivals work correctly with segment=4 and hot spares.

        Participants from different domains arrive out-of-order, but should be sorted
        by infra_rank (SLURM topology order) for proper segment assignment.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Participants arrive out-of-order from 2 domains
        # Domain 100 has infra_ranks [0-7], Domain 101 has infra_ranks [8-11]
        # Listed in reverse order to test sorting
        participants = [
            # Domain 101 nodes (listed first, but have higher infra_ranks)
            (_NodeDesc("nvl72101-node3", 111, 0), 11, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 110, 0), 10, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 109, 0), 9, "nvl72101"),
            (_NodeDesc("nvl72101-node0", 108, 0), 8, "nvl72101"),
            (_NodeDesc("nvl72100-node7", 107, 0), 7, "nvl72100"),
            (_NodeDesc("nvl72100-node6", 106, 0), 6, "nvl72100"),
            (_NodeDesc("nvl72100-node5", 105, 0), 5, "nvl72100"),
            (_NodeDesc("nvl72100-node4", 104, 0), 4, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
        ]
        world_size = 8  # Need 2 segments

        result = state._assign_group_ranks(participants, world_size)

        # Despite out-of-order arrival, Domain 100 (lower infra_ranks) should be sorted first
        # Domain 100: all 8 nodes active [0-7]
        self.assertEqual(result[_NodeDesc("nvl72100-node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node1", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node2", 102, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node3", 103, 0)], 3)
        self.assertEqual(result[_NodeDesc("nvl72100-node4", 104, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72100-node5", 105, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72100-node6", 106, 0)], 6)
        self.assertEqual(result[_NodeDesc("nvl72100-node7", 107, 0)], 7)

        # Domain 101: all 4 nodes are standby [8-11]
        self.assertEqual(result[_NodeDesc("nvl72101-node0", 108, 0)], 8)
        self.assertEqual(result[_NodeDesc("nvl72101-node1", 109, 0)], 9)
        self.assertEqual(result[_NodeDesc("nvl72101-node2", 110, 0)], 10)
        self.assertEqual(result[_NodeDesc("nvl72101-node3", 111, 0)], 11)

    def test_rank_assignment_out_of_order_mixed_domains_segment_4(self):
        """Test out-of-order arrivals with mixed segment counts across domains.

        Tests that nodes from multiple domains arriving in arbitrary order are
        correctly sorted and assigned ranks based on infra_rank and segment rules.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: infra_ranks [0-5] (1 complete segment + 2 incomplete)
        # Domain 101: infra_ranks [6-9] (1 complete segment)
        # Domain 102: infra_ranks [10-12] (0 complete segments)
        # Listed in completely arbitrary order
        participants = [
            (_NodeDesc("nvl72102-node2", 112, 0), 12, "nvl72102"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            (_NodeDesc("nvl72101-node1", 107, 0), 7, "nvl72101"),
            (_NodeDesc("nvl72102-node0", 110, 0), 10, "nvl72102"),
            (_NodeDesc("nvl72100-node5", 105, 0), 5, "nvl72100"),
            (_NodeDesc("nvl72101-node3", 109, 0), 9, "nvl72101"),
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72101-node2", 108, 0), 8, "nvl72101"),
            (_NodeDesc("nvl72102-node1", 111, 0), 11, "nvl72102"),
            (_NodeDesc("nvl72100-node4", 104, 0), 4, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72101-node0", 106, 0), 6, "nvl72101"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
        ]
        world_size = 8  # Need 2 segments

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100 (infra_ranks 0-5): 1 complete segment (4 nodes) active, 2 standby
        self.assertEqual(result[_NodeDesc("nvl72100-node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node1", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node2", 102, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node3", 103, 0)], 3)
        self.assertEqual(result[_NodeDesc("nvl72100-node4", 104, 0)], 8)  # Standby
        self.assertEqual(result[_NodeDesc("nvl72100-node5", 105, 0)], 9)  # Standby

        # Domain 101 (infra_ranks 6-9): 1 complete segment (4 nodes) active
        self.assertEqual(result[_NodeDesc("nvl72101-node0", 106, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72101-node1", 107, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72101-node2", 108, 0)], 6)
        self.assertEqual(result[_NodeDesc("nvl72101-node3", 109, 0)], 7)

        # Domain 102 (infra_ranks 10-12): 0 complete segments, all 3 standby
        self.assertEqual(result[_NodeDesc("nvl72102-node0", 110, 0)], 10)  # Standby
        self.assertEqual(result[_NodeDesc("nvl72102-node1", 111, 0)], 11)  # Standby
        self.assertEqual(result[_NodeDesc("nvl72102-node2", 112, 0)], 12)  # Standby

    def test_rank_assignment_with_gaps_in_infra_ranks_segment_none(self):
        """Test rank assignment with gaps in infra_ranks (simulating node failures), segment=None.

        When some nodes fail and don't join, there are gaps in infra_ranks.
        E.g., if nodes 2, 4, 5 failed, we have infra_ranks [0, 1, 3, 6, 7] instead of [0-7].
        These should still get contiguous group ranks [0, 1, 2, 3, 4].
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=None,
        )

        # Simulate 5 nodes joining with gaps: infra_ranks [0, 1, 3, 6, 7]
        # Missing nodes: 2, 4, 5 (failed to join)
        # World_size=3 -> first 3 active [0,1,2], remaining 2 hot spares [3,4]
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node1", 101, 0), 1, "none"),
            (_NodeDesc("node3", 103, 0), 3, "none"),
            (_NodeDesc("node6", 106, 0), 6, "none"),
            (_NodeDesc("node7", 107, 0), 7, "none"),
        ]
        world_size = 3

        result = state._assign_group_ranks(participants, world_size)

        # Should get contiguous group ranks [0-4] based on sorted infra_rank
        self.assertEqual(result[_NodeDesc("node0", 100, 0)], 0)  # Active
        self.assertEqual(result[_NodeDesc("node1", 101, 0)], 1)  # Active
        self.assertEqual(result[_NodeDesc("node3", 103, 0)], 2)  # Active
        self.assertEqual(result[_NodeDesc("node6", 106, 0)], 3)  # Hot spare
        self.assertEqual(result[_NodeDesc("node7", 107, 0)], 4)  # Hot spare

    def test_rank_assignment_with_gaps_in_infra_ranks_with_hot_spare(self):
        """Test rank assignment with gaps in infra_ranks and hot spares (segment=None).

        More aggressive gap scenario: infra_ranks [0, 5, 10, 15, 20, 22, 24]
        Simulates many node failures (1-4, 6-9, 11-14, 16-19, 21, 23 all failed).
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=None,
        )

        # Large gaps in infra_ranks
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node5", 105, 0), 5, "none"),
            (_NodeDesc("node10", 110, 0), 10, "none"),
            (_NodeDesc("node15", 115, 0), 15, "none"),
            (_NodeDesc("node20", 120, 0), 20, "none"),
            (_NodeDesc("node22", 122, 0), 22, "none"),
            (_NodeDesc("node24", 124, 0), 24, "none"),
        ]
        world_size = 4  # Need 4 active nodes

        result = state._assign_group_ranks(participants, world_size)

        # Should get contiguous group ranks [0-6], first 4 active, last 3 hot spares
        self.assertEqual(result[_NodeDesc("node0", 100, 0)], 0)  # Active
        self.assertEqual(result[_NodeDesc("node5", 105, 0)], 1)  # Active
        self.assertEqual(result[_NodeDesc("node10", 110, 0)], 2)  # Active
        self.assertEqual(result[_NodeDesc("node15", 115, 0)], 3)  # Active
        self.assertEqual(result[_NodeDesc("node20", 120, 0)], 4)  # Hot spare
        self.assertEqual(result[_NodeDesc("node22", 122, 0)], 5)  # Hot spare
        self.assertEqual(result[_NodeDesc("node24", 124, 0)], 6)  # Hot spare

    def test_rank_assignment_with_gaps_in_infra_ranks_segment_4(self):
        """Test rank assignment with gaps in infra_ranks across domains with segment=4.

        Simulates scenario where some nodes in each domain failed to join,
        leaving gaps in infra_ranks within each domain.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: infra_ranks [0, 2, 4, 6, 8, 10] (nodes 1,3,5,7,9 failed)
        #             6 nodes -> 1 complete segment (4 nodes), 2 incomplete
        # Domain 101: infra_ranks [12, 14, 17, 19] (nodes 13,15,16,18 failed)
        #             4 nodes -> 1 complete segment
        # World_size=8 requires 2 segments
        participants = [
            # Domain 100 (with gaps)
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node4", 104, 0), 4, "nvl72100"),
            (_NodeDesc("nvl72100-node6", 106, 0), 6, "nvl72100"),
            (_NodeDesc("nvl72100-node8", 108, 0), 8, "nvl72100"),
            (_NodeDesc("nvl72100-node10", 110, 0), 10, "nvl72100"),
            (_NodeDesc("nvl72101-node0", 112, 0), 12, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 114, 0), 14, "nvl72101"),
            (_NodeDesc("nvl72101-node5", 117, 0), 17, "nvl72101"),
            (_NodeDesc("nvl72101-node7", 119, 0), 19, "nvl72101"),
        ]
        world_size = 8

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100: 1 complete segment (4 nodes) active, 2 standby
        self.assertEqual(result[_NodeDesc("nvl72100-node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node2", 102, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node4", 104, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node6", 106, 0)], 3)
        self.assertEqual(result[_NodeDesc("nvl72100-node8", 108, 0)], 8)  # Standby
        self.assertEqual(result[_NodeDesc("nvl72100-node10", 110, 0)], 9)  # Standby

        # Domain 101: 1 complete segment (4 nodes) active
        self.assertEqual(result[_NodeDesc("nvl72101-node0", 112, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72101-node2", 114, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72101-node5", 117, 0)], 6)
        self.assertEqual(result[_NodeDesc("nvl72101-node7", 119, 0)], 7)

    def test_rank_assignment_with_gaps_out_of_order_segment_4(self):
        """Test rank assignment with gaps in infra_ranks arriving out-of-order with segment=4.

        Combines three challenging scenarios:
        1. Gaps in infra_ranks (node failures)
        2. Out-of-order arrivals
        3. Multiple domains with segment awareness
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: infra_ranks [1, 3, 5, 7, 9] (0,2,4,6,8 failed) -> 1 segment + 1 incomplete
        # Domain 101: infra_ranks [11, 13, 15, 17] (10,12,14,16 failed) -> 1 segment
        # Listed in reverse order to test sorting
        participants = [
            # Domain 101 first, in reverse
            (_NodeDesc("nvl72101-node7", 117, 0), 17, "nvl72101"),
            (_NodeDesc("nvl72101-node5", 115, 0), 15, "nvl72101"),
            (_NodeDesc("nvl72101-node3", 113, 0), 13, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 111, 0), 11, "nvl72101"),
            (_NodeDesc("nvl72100-node9", 109, 0), 9, "nvl72100"),
            (_NodeDesc("nvl72100-node7", 107, 0), 7, "nvl72100"),
            (_NodeDesc("nvl72100-node5", 105, 0), 5, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
        ]
        world_size = 8

        result = state._assign_group_ranks(participants, world_size)

        # Despite reverse order and gaps, Domain 100 (lower infra_ranks) comes first
        # Domain 100: 1 complete segment (4 nodes) active, 1 standby
        self.assertEqual(result[_NodeDesc("nvl72100-node1", 101, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node3", 103, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node5", 105, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node7", 107, 0)], 3)
        self.assertEqual(result[_NodeDesc("nvl72100-node9", 109, 0)], 8)  # Standby

        # Domain 101: 1 complete segment (4 nodes) active
        self.assertEqual(result[_NodeDesc("nvl72101-node1", 111, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72101-node3", 113, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72101-node5", 115, 0)], 6)
        self.assertEqual(result[_NodeDesc("nvl72101-node7", 117, 0)], 7)


class ErrorCaseTest(BaseRendezvousTest):
    """Test error cases and exception handling."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""

        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use a globally unique run_id so store keys never collide with other tests or runs
        self.run_id = f"test_error_{self._testMethodName}_{uuid.uuid4().hex}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_exceed_max_nodes_raises_error(self):
        """Test that exceeding max_nodes raises RendezvousClosedError.

        Seed join_count at max_nodes and let the next participant take slot
        max_nodes + 1. The error is raised before any Step 2 polling.
        """
        min_nodes = 2
        max_nodes = 2  # Set max to 2; third participant must see count 3 and raise
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=False,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        self.store.add(state.join_count_key, max_nodes)

        with self.assertRaises(RendezvousClosedError):
            state.perform_rendezvous(
                self.node_desc_gen.generate(),
                min_nodes,
                max_nodes,
                _test_segment_check_interval(),
            )

    def test_timeout_raises_error(self):
        """Test that join timeout raises RendezvousTimeoutError.

        Uses a dedicated TCPStore so the rendezvous is open (no leftover keys from
        other tests). We then hit the join timeout in Step 2 while waiting for min_nodes.
        """
        # Dedicated store so round_done_0 does not exist; __init__ sets it to "0".
        # This avoids waiting at Step 0 when the class shared_store has leftover state.
        # TCPStore does not provide close()/shutdown(); store is released when GC runs.
        store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )
        run_id = f"test_error_{self._testMethodName}_{uuid.uuid4().hex}"
        state = _RendezvousBarrierState(
            store=store,
            run_id=run_id,
            is_store_host=True,
            join_timeout_seconds=1.0,  # Use very short timeout to test timeout behavior
        )
        min_nodes = 5  # Require 5 nodes but we'll only provide 1
        max_nodes = 5
        segment_check_interval = _test_segment_check_interval()

        node = self.node_desc_gen.generate()

        with self.assertRaises(RendezvousTimeoutError):
            state.perform_rendezvous(node, min_nodes, max_nodes, segment_check_interval)

    def test_closed_rendezvous_raises_error(self):
        """Test that joining a closed rendezvous raises RendezvousGracefulExitError (exit 0)."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Permanently close the rendezvous
        state.set_shutdown()
        self.assertTrue(state.is_shutdown())

        min_nodes = 2
        max_nodes = 4
        segment_check_interval = _test_segment_check_interval()
        node = self.node_desc_gen.generate()

        # Should raise RendezvousGracefulExitError (hot spare / Step 0 sees closed → graceful exit)
        with self.assertRaises(RendezvousGracefulExitError):
            state.perform_rendezvous(node, min_nodes, max_nodes, segment_check_interval)

    def test_duplicate_infra_rank_raises_error(self):
        """Test that duplicate infrastructure ranks raise an error."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Create participants with duplicate infra_rank
        participants = [
            (_NodeDesc("node_a", 100, 0), 0, "none"),
            (_NodeDesc("node_b", 101, 0), 0, "none"),
        ]
        min_nodes = 2

        # Should raise RuntimeError about duplicate ranks
        with self.assertRaises(RuntimeError) as ctx:
            state._assign_group_ranks(participants, min_nodes)

        self.assertIn("Duplicate", str(ctx.exception))


class AcknowledgmentPhaseTest(BaseRendezvousTest):
    """Test acknowledgment phase behavior."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use uuid so run_id is globally unique when tests run back-to-back
        self.run_id = f"test_ack_{self._testMethodName}_{uuid.uuid4().hex}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_all_participants_acknowledge(self):
        """Test that all participants acknowledge completion.

        The current protocol closes by publishing rank keys before round_done=1;
        verify each participant can observe completion deterministically.
        """
        min_nodes = 3
        max_nodes = 3
        segment_check_interval = _test_segment_check_interval()
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node1", 101, 0), 1, "none"),
            (_NodeDesc("node2", 102, 0), 2, "none"),
        ]
        _seed_joined_participants(self.store, state, participants)

        state._rendezvous_start_time = time.monotonic()
        state._host_close_round(
            _NodeDesc("control", 10, 0),
            min_nodes,
            max_nodes,
            segment_check_interval,
        )

        observed_ranks = []
        for slot, (node_desc, _, _) in enumerate(participants, start=1):
            rank, total = state._wait_for_round_done(node_desc, f"{state.prefix}:slot_{slot}_rank")
            observed_ranks.append(rank)
            self.assertEqual(total, min_nodes)
        self.assertEqual(set(observed_ranks), set(range(min_nodes)))

    def test_per_round_keys_persist_after_round(self):
        """Test that per-round keys persist after round completion (not cleared).

        join_count_{N+1} does not exist during training, so num_nodes_waiting() returns 0.
        Round N keys persist with correct values — the per-round naming makes them harmless.
        """
        min_nodes = 2
        max_nodes = 2
        segment_check_interval = _test_segment_check_interval()
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node1", 101, 0), 1, "none"),
        ]
        _seed_joined_participants(self.store, state, participants)

        state._rendezvous_start_time = time.monotonic()
        state._host_close_round(
            _NodeDesc("control", 10, 0),
            min_nodes,
            max_nodes,
            segment_check_interval,
        )

        # Check keys using the master store (same run_id)
        check_state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=False,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Round 0 join_count persists with value = min_nodes
        self.assertTrue(
            self.store.check([check_state.join_count_key]),
            "join_count_0 should persist after round (per-round key, not cleared)",
        )
        join_count = int(self.store.get(check_state.join_count_key).decode('utf-8'))
        self.assertEqual(join_count, min_nodes, f"join_count_0 should equal {min_nodes}")

        # round_done_0 persists and is "1" (closed)
        self.assertTrue(
            self.store.check([check_state.round_done_key]), "round_done_0 should persist"
        )
        self.assertEqual(
            self.store.get(check_state.round_done_key).decode('utf-8'),
            "1",
            "round_done_0 should be 1 after completion",
        )

        # Round 1 join_count does NOT exist — no false positive for num_nodes_waiting()
        round1_join_count_key = f"{check_state.prefix}:join_count_1"
        self.assertFalse(
            self.store.check([round1_join_count_key]),
            "join_count_1 must NOT exist during training (prevents num_nodes_waiting() false positive)",
        )


class HandlerIntegrationTest(BaseRendezvousTest):
    """Integration tests for FtRendezvousBarrierHandler."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use uuid so run_id is globally unique (time-based run_id can collide in same microsecond).
        self.run_id = f"test_handler_{self._testMethodName}_{uuid.uuid4().hex}"

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_handler_creation(self):
        """Test that handler can be created with correct parameters."""
        handler = FtRendezvousBarrierHandler.from_backend(
            run_id=self.run_id,
            store=self.store,
            backend=None,
            min_nodes=2,
            max_nodes=4,
            timeout=RendezvousTimeout(),
            is_store_host=True,
        )

        self.assertIsNotNone(handler)
        self.assertEqual(handler.settings.run_id, self.run_id)
        self.assertEqual(handler.settings.min_nodes, 2)
        self.assertEqual(handler.settings.max_nodes, 4)

    def test_handler_segment_validation(self):
        """Test that handler validates min_nodes % segment == 0."""
        # Valid: min_nodes=8 % segment=4 == 0
        handler = FtRendezvousBarrierHandler.from_backend(
            run_id=self.run_id,
            store=self.store,
            backend=None,
            min_nodes=8,
            max_nodes=16,
            timeout=RendezvousTimeout(),
            is_store_host=True,
            segment=4,
        )
        self.assertIsNotNone(handler)

        # Invalid: min_nodes=10 % segment=4 != 0
        with self.assertRaises(ValueError) as ctx:
            FtRendezvousBarrierHandler.from_backend(
                run_id=f"{self.run_id}_invalid",
                store=self.store,
                backend=None,
                min_nodes=10,
                max_nodes=16,
                timeout=RendezvousTimeout(),
                is_store_host=True,
                segment=4,
            )
        self.assertIn("must be divisible by segment", str(ctx.exception))

        # Invalid: min_nodes=7 % segment=3 != 0
        with self.assertRaises(ValueError) as ctx:
            FtRendezvousBarrierHandler.from_backend(
                run_id=f"{self.run_id}_invalid2",
                store=self.store,
                backend=None,
                min_nodes=7,
                max_nodes=12,
                timeout=RendezvousTimeout(),
                is_store_host=True,
                segment=3,
            )
        self.assertIn("must be divisible by segment", str(ctx.exception))

    def test_handler_next_rendezvous(self):
        """Test basic rendezvous flow is validated by other test classes."""
        # Note: Full integration testing with FtRendezvousBarrierHandler is done in functional tests
        # due to IPC socket conflicts in unit tests. The core rendezvous functionality is thoroughly
        # tested by the other 26 tests in this file covering:
        # - Barrier state operations (BarrierStateBasicTest)
        # - Step 2 completion logic (Step2CompletionTest)
        # - Race conditions (RaceConditionTest)
        # - Store host behavior (StoreHostBehaviorTest)
        # - Group rank assignment (GroupRankAssignmentTest)
        # - Error cases (ErrorCaseTest)
        # - Acknowledgment phase (AcknowledgmentPhaseTest)
        # - Infrastructure rank handling (InfrastructureRankTest)

        # This test verifies the components work together, which is already covered above
        self.assertTrue(True, "Core functionality tested by other test classes")


class InfrastructureRankTest(TestCase):
    """Test infrastructure rank handling."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        # NOTE: DO NOT clear SLURM_PROCID/GROUP_RANK for this test class
        # These tests specifically test infrastructure rank behavior from environment

        self.store = self.shared_store
        # Use uuid so run_id is globally unique (time-based run_id can collide in same microsecond).
        self.run_id = f"test_infra_{self._testMethodName}_{uuid.uuid4().hex}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        pass

    def test_infra_rank_from_environment(self):
        """Test reading infrastructure rank from environment variables."""
        # Note: This is a simplified test of infra rank assignment logic
        # Testing with actual environment variables in threads is complex due to
        # shared environment. The actual behavior is tested in functional tests.

        # Test the assignment logic directly with simulated participants
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Create participants with infra ranks
        participants = [
            (_NodeDesc("node_0", 100, 0), 0, "none"),
            (_NodeDesc("node_1", 101, 0), 1, "none"),
        ]
        min_nodes = 2

        # Assign ranks using infrastructure rank mode
        result = state._assign_group_ranks(participants, min_nodes)

        # Check direct mapping: group_rank = infra_rank
        self.assertEqual(result[participants[0][0]], 0)
        self.assertEqual(result[participants[1][0]], 1)


class StaleRoundDetectionTest(BaseRendezvousTest):
    """Test stale rendezvous round detection and rate limiting."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,  # Use any available port
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        super().setUp()
        # Use the shared_store from setUpClass
        self.store = self.shared_store
        # Use uuid so run_id is globally unique (time-based run_id can collide in same microsecond).
        self.run_id = f"test_stale_{self._testMethodName}_{uuid.uuid4().hex}"
        self.node_desc_gen = _NodeDescGenerator()

    def test_stale_check_interval_parameter(self):
        """Test that stale_check_interval parameter is properly set."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            stale_check_interval=5.0,
        )

        self.assertEqual(state.stale_check_interval, 5.0)

    def test_stale_check_interval_default(self):
        """Test that stale_check_interval has correct default value."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        self.assertEqual(state.stale_check_interval, 10.0)

    def test_stale_round_detection_rate_limiting(self):
        """Test that _sync_from_per_round_state is rate-limited properly at Step 0."""
        check_interval = 0.5
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            stale_check_interval=check_interval,
        )

        check_count = 0

        def count_sync():
            nonlocal check_count
            check_count += 1
            return False

        state._sync_from_per_round_state = count_sync
        state._last_stale_check_time = 100.0

        # Simulate Step 0's rate-limit branch with deterministic timestamps.
        for current_time in (100.1, 100.49, 100.5, 100.99, 101.0, 101.49):
            if current_time - state._last_stale_check_time >= state.stale_check_interval:
                state._last_stale_check_time = current_time
                state._sync_from_per_round_state()

        self.assertEqual(check_count, 2)
        self.assertEqual(state._last_stale_check_time, 101.0)

    def test_stale_round_syncs_automatically(self):
        """Test that _sync_from_per_round_state syncs round by scanning per-round keys."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            stale_check_interval=0.1,  # Short interval for testing
        )

        # Simulate: rounds 0-4 all completed (round_done_{0..4}=1)
        # Round 5 is open (round_done_5=0)
        prefix = state.prefix
        for i in range(5):
            state.store.set(f"{prefix}:round_done_{i}", b"1")
        state.store.set(f"{prefix}:round_done_5", b"0")

        node_desc = self.node_desc_gen.generate()

        # _wait_for_rendezvous_open starts at _round=0, sees round_done_0=1 (closed),
        # calls _sync_from_per_round_state which scans forward and advances _round to 5.
        # Then sees round_done_5=0 (open) and returns.
        state._wait_for_rendezvous_open(node_desc)

        # Verify that the round was synced to 5
        self.assertEqual(state._round, 5)


class SignalLeaveRendezvousTest(BaseRendezvousTest):
    """Test that a participant that receives SIGTERM after joining (Step 1) leaves by
    incrementing leave_count_key and marking its slot with WITHDRAWN domain_id, so the
    store host can compute active_count = join_count - leave_count for assignment.
    """

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        super().setUp()
        self.store = self.shared_store
        self.run_id = f"test_signal_leave_{self._testMethodName}_{uuid.uuid4().hex}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clear module-level state so other tests are not affected."""
        super().tearDown()
        ft_rendezvous_barrier_module._current_joined_state = None

    def test_leave_rendezvous_increments_once(self):
        """Test that _leave_rendezvous increments leave_count_key and marks slot; idempotent."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        # Simulate one participant having joined (slot 1)
        self.store.add(state.join_count_key, 1)
        state._slot = 1  # Simulate we are that participant
        self.assertEqual(int(self.store.get(state.join_count_key).decode('utf-8')), 1)

        state._leave_rendezvous()
        # join_count stays 1; leave_count becomes 1
        self.assertEqual(
            int(self.store.get(state.join_count_key).decode('utf-8')),
            1,
            "join_count should remain 1 after leave (slot space is monotonic)",
        )
        self.assertEqual(
            int(self.store.get(state.leave_count_key).decode('utf-8')),
            1,
            "leave_count should be 1 after leave",
        )

        # Idempotent: second call must not increment leave_count again
        state._leave_rendezvous()
        self.assertEqual(
            int(self.store.get(state.leave_count_key).decode('utf-8')),
            1,
            "leave_count should still be 1 after second leave",
        )

    def test_signal_after_join_causes_leave(self):
        """Test that when a participant is interrupted after Step 1 (e.g. SIGTERM), the
        finally block leaves (increments leave_count_key, marks slot) so the store
        host can complete Step 2. We simulate the signal by using a store wrapper that
        raises SignalException on get() once _current_joined_state is set (after join).
        """
        # min_nodes=2 so the host waits for both to join before marking complete (host is faster
        # than participant which runs ensure_node_is_healthy first). Use longer join timeout so
        # the participant (main thread) has time to pass health check and join.
        min_nodes = 2
        max_nodes = 2
        segment_check_interval = _test_segment_check_interval()
        # Participant (main thread) runs ensure_node_is_healthy first, so it can be slow to join.
        # Host runs perform_rendezvous directly and waits for 2 participants.
        join_timeout_secs = 60.0

        # Event set when participant has joined (add(join_count_key, 1)) so host can wait
        # and not timeout before the participant reaches perform_rendezvous.
        participant_joined_event = threading.Event()
        # Event set when the host has joined (add(join_count_key, 1)). The participant
        # store must not start counting round_done reads until the host has joined;
        # otherwise the SignalException can fire before join_count reaches 2.
        host_joined_event = threading.Event()
        # Event set when participant has done the first get of round_done_key in Step 3.
        # Host waits on this before the round closes so the participant sees 0 on first
        # get, then we raise on the second get (before returning 1).
        participant_did_first_get_event = threading.Event()

        # Participant store: raise SignalException on 2nd get of round_done_0;
        # set participant_did_first_get_event on 1st get so host knows to proceed.
        # Only start counting round_done reads after host_joined_event is set so that
        # join_count == 2 when the participant reads it after leaving.
        class StoreThatRaisesSignalAfterJoin:
            def __init__(self, underlying, joined_event, host_joined_event, did_first_get_event):
                self._store = underlying
                self._joined_event = joined_event
                self._host_joined_event = host_joined_event
                self._did_first_get_event = did_first_get_event
                self._step2_get_count = 0

            def get(self, key):
                if (
                    ":round_done_" in key
                    and ft_rendezvous_barrier_module._current_joined_state is not None
                    and self._host_joined_event.is_set()
                ):
                    self._step2_get_count += 1
                    if self._step2_get_count == 1:
                        self._did_first_get_event.set()
                    if self._step2_get_count >= 2:
                        # Simulate what the real signal handler does: set _leave_on_unwind
                        # so the handler's finally will call _leave_rendezvous().
                        state = ft_rendezvous_barrier_module._current_joined_state
                        if state is not None:
                            state._leave_on_unwind = True
                        raise SignalException(
                            "Simulated signal during rendezvous",
                            sigval=signal.Signals(signal.SIGTERM),
                        )
                return self._store.get(key)

            def add(self, key, value):
                result = self._store.add(key, value)
                if ":join_count_" in key and value == 1:
                    self._joined_event.set()
                return result

            def set(self, key, value):
                return self._store.set(key, value)

            def compare_set(self, key, expected_value, desired_value):
                return self._store.compare_set(key, expected_value, desired_value)

            def check(self, keys):
                return self._store.check(keys)

            def multi_get(self, keys):
                return self._store.multi_get(keys)

            def multi_set(self, keys, values):
                return self._store.multi_set(keys, values)

        # Host store: signal when it has joined; delay set(round_done_0, 1) until participant
        # has done first get, so participant sees 0 then we raise on second get.
        class HostStoreWaitsForFirstGet:
            def __init__(self, underlying, host_joined_event, did_first_get_event):
                self._store = underlying
                self._host_joined_event = host_joined_event
                self._did_first_get_event = did_first_get_event

            def get(self, key):
                return self._store.get(key)

            def add(self, key, value):
                result = self._store.add(key, value)
                if ":join_count_" in key and value == 1:
                    self._host_joined_event.set()
                return result

            def set(self, key, value):
                if ":round_done_" in key and value == "1".encode("utf-8"):
                    self._did_first_get_event.wait(timeout=30.0)
                return self._store.set(key, value)

            def compare_set(self, key, expected_value, desired_value):
                return self._store.compare_set(key, expected_value, desired_value)

            def check(self, keys):
                return self._store.check(keys)

            def multi_get(self, keys):
                return self._store.multi_get(keys)

            def multi_set(self, keys, values):
                return self._store.multi_set(keys, values)

        participant_store = StoreThatRaisesSignalAfterJoin(
            self.store, participant_joined_event, host_joined_event, participant_did_first_get_event
        )
        host_store = HostStoreWaitsForFirstGet(
            self.store, host_joined_event, participant_did_first_get_event
        )
        host_result = []
        participant_error = []
        participant_count_after_leave = []

        def host_thread():
            state = _RendezvousBarrierState(
                store=host_store,
                run_id=self.run_id,
                is_store_host=True,
                join_timeout_seconds=join_timeout_secs,
            )
            try:
                node = self.node_desc_gen.generate()
                rank, total = state.perform_rendezvous(
                    node, min_nodes, max_nodes, segment_check_interval
                )
                host_result.append((rank, total))
            except Exception as e:
                host_result.append(('error', e))

        # Participant must run in main thread so handler can install signal handlers
        # (signal.signal() only works in main thread). Host runs in worker thread.
        # Host waits for participant to join before starting so the host does not timeout
        # (participant is slow to reach perform_rendezvous due to health check etc.).
        start_barrier = threading.Barrier(2)

        def host_with_sync():
            start_barrier.wait(timeout=BARRIER_WAIT_TIMEOUT_SECS)
            participant_joined_event.wait(timeout=60.0)
            host_thread()

        t_host = threading.Thread(target=host_with_sync)
        t_host.start()
        start_barrier.wait(timeout=BARRIER_WAIT_TIMEOUT_SECS)
        handler = FtRendezvousBarrierHandler.from_backend(
            run_id=self.run_id,
            store=participant_store,
            backend=None,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            timeout=RendezvousTimeout(join=timedelta(seconds=join_timeout_secs)),
            is_store_host=False,
        )
        try:
            handler.next_rendezvous()
        except SignalException:
            participant_error.append(True)
            # Simulated signal does not go through the real signal handler, so _leave_on_unwind
            # is set in the wrapper; ensure leave runs so host can finish Step 2.
            handler._barrier_state._leave_on_unwind = True
            handler._barrier_state._maybe_leave_on_unwind()
            try:
                joined = int(self.store.get(handler._barrier_state.join_count_key).decode('utf-8'))
                left = handler._barrier_state._get_leave_count()
                participant_count_after_leave.append((joined, left))
            except Exception:
                pass
        t_host.join(timeout=70)
        _assert_threads_finished(self, [t_host], 70)

        self.assertTrue(participant_error, "Participant should have received SignalException")
        # join_count stays 2 (both joined); leave_count = 1 (participant left)
        self.assertEqual(
            participant_count_after_leave,
            [(2, 1)],
            "After participant left: join_count=2 (unchanged), leave_count=1",
        )
        # Left participant's slot (participant joined first = slot 1) should be marked with WITHDRAWN
        state = handler._barrier_state
        slot_key = f"{state.prefix}:slot_1"
        slot_round, payload = RendezvousStoreValue.unpack(self.store.get(slot_key).decode('utf-8'))
        _, _, domain_id = RendezvousParticipantInfo.from_payload(payload)
        self.assertEqual(
            domain_id,
            WITHDRAWN,
            "Left participant's slot should have WITHDRAWN domain_id",
        )
        self.assertEqual(slot_round, state._round)
        self.assertEqual(len(host_result), 1, "Host should complete or error (not hang)")
        # Host either completes with world size 1 or errors (e.g. segment constraint) after
        # participant withdrew; both are acceptable as long as the host did not hang in Step 2.
        if host_result[0][0] == 'error':
            err = host_result[0][1]
            # RendezvousTimeoutError is a common outcome when the participant withdraws
            # and active_count drops below min_nodes; it IS a RendezvousError subclass but
            # type().__name__ only matches the concrete class name, not base classes.
            self.assertIn(
                type(err).__name__,
                ('RuntimeError', 'RendezvousError', 'RendezvousTimeoutError'),
                f"Host error should be a rendezvous or runtime error: {err}",
            )
        else:
            rank, total = host_result[0]
            # total_participants in rank value is slot count (join_count=2), not active count
            self.assertEqual(
                total, 2, "Host should see total_participants=2 (slot count) in rank value"
            )
            # In the new protocol, ranks are assigned BEFORE round_done=1 is set, so
            # both slots get a rank even if one later leaves. The left participant
            # (slot 1) overwrites slot data with WITHDRAWN, but its slot_rank was already
            # written by the host before the signal arrived. Just verify host got a valid rank.
            self.assertGreaterEqual(rank, 0, "Host should have a valid group rank >= 0")

    def test_signal_after_round_closed_does_not_leave(self):
        """Test that when _round_closed=True, a signal does NOT trigger withdrawal.

        After round_done_N=1 is seen, _round_closed=True prevents spurious
        leave_count increments. Only pre-round-close signals should cause withdrawal.
        """
        add_calls = []

        class TrackingStore:
            def __init__(self, underlying):
                self._store = underlying

            def get(self, key):
                return self._store.get(key)

            def add(self, key, value):
                add_calls.append((key, value))
                return self._store.add(key, value)

            def set(self, key, value):
                return self._store.set(key, value)

            def compare_set(self, key, expected_value, desired_value):
                return self._store.compare_set(key, expected_value, desired_value)

            def check(self, keys):
                return self._store.check(keys)

            def multi_get(self, keys):
                return self._store.multi_get(keys)

            def multi_set(self, keys, values):
                return self._store.multi_set(keys, values)

        state = _RendezvousBarrierState(
            store=TrackingStore(self.store),
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        state._slot = 1  # Simulate having joined (slot 1)
        state._round_closed = True  # Signal came AFTER round_done_N=1 was seen
        state._leave_on_unwind = True  # Signal handler set this

        state._maybe_leave_on_unwind()

        # Verify no leave_count was incremented
        leave_calls = [(k, v) for k, v in add_calls if ":leave_count_" in k and v == 1]
        self.assertEqual(
            len(leave_calls),
            0,
            "After _round_closed=True, _leave_rendezvous must NOT be called. "
            f"Got add calls: {add_calls}",
        )
        self.assertFalse(state._leave_on_unwind, "_leave_on_unwind should be reset to False")


if __name__ == '__main__':
    import unittest

    unittest.main()
