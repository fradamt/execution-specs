"""
Ethereum Specification
^^^^^^^^^^^^^^^^^^^^^^

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

Entry point for the Ethereum specification.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from ethereum_rlp import rlp
from ethereum_types.bytes import Bytes, Bytes0
from ethereum_types.numeric import U64, U256, Uint

from ethereum.crypto.hash import Hash32, keccak256
from ethereum.exceptions import (
    EthereumException,
    InvalidBlock,
)

from . import vm
from .blocks import Block, Header, Log, Receipt, Withdrawal, encode_receipt
from .bloom import logs_bloom
from .fork_types import Account, Address, Authorization, VersionedHash
from .requests import (
    CONSOLIDATION_REQUEST_TYPE,
    DEPOSIT_REQUEST_TYPE,
    WITHDRAWAL_REQUEST_TYPE,
    compute_requests_hash,
    parse_deposit_requests,
)
from .state import (
    State,
    TransientStorage,
    account_exists_and_is_empty,
    destroy_account,
    get_account,
    increment_nonce,
    modify_state,
    set_account_balance,
    state_root,
    begin_transaction,
    commit_transaction,
    rollback_transaction,
)
from .transactions import (
    AccessListTransaction,
    BlobTransaction,
    FeeMarketTransaction,
    LegacyTransaction,
    SetCodeTransaction,
    Transaction,
    decode_transaction,
    encode_transaction,
    get_transaction_hash,
    recover_sender,
    calculate_intrinsic_cost,
)
from .trie import root, trie_set, Trie
from .utils.hexadecimal import hex_to_address
from .utils.message import prepare_message
from .vm import Message
from .vm.eoa_delegation import is_valid_delegation
from .vm.gas import (
    calculate_blob_gas_price,
    calculate_excess_blob_gas,
    calculate_total_blob_gas,
)
from .vm.interpreter import MessageCallOutput, process_message_call, MAX_CODE_SIZE

BASE_FEE_MAX_CHANGE_DENOMINATOR = Uint(8)
ELASTICITY_MULTIPLIER = Uint(2)
GAS_LIMIT_ADJUSTMENT_FACTOR = Uint(1024)
GAS_LIMIT_MINIMUM = Uint(5000)
EMPTY_OMMER_HASH = keccak256(rlp.encode([]))
SYSTEM_ADDRESS = hex_to_address("0xfffffffffffffffffffffffffffffffffffffffe")
BEACON_ROOTS_ADDRESS = hex_to_address(
    "0x000F3df6D732807Ef1319fB7B8bB8522d0Beac02"
)
SYSTEM_TRANSACTION_GAS = Uint(30000000)
MAX_BLOB_GAS_PER_BLOCK = U64(1179648)
VERSIONED_HASH_VERSION_KZG = b"\x01"

WITHDRAWAL_REQUEST_PREDEPLOY_ADDRESS = hex_to_address(
    "0x00000961Ef480Eb55e80D19ad83579A64c007002"
)
CONSOLIDATION_REQUEST_PREDEPLOY_ADDRESS = hex_to_address(
    "0x0000BBdDc7CE488642fb579F8B00f3a590007251"
)
HISTORY_STORAGE_ADDRESS = hex_to_address(
    "0x0000F90827F1C53a10cb7A02335B175320002935"
)
HISTORY_SERVE_WINDOW = 8192


@dataclass
class BlockChain:
    """
    History and current state of the block chain.
    """

    blocks: List[Block]
    state: State
    chain_id: U64
    last_transactions_root: Root
    last_receipt_root: Root
    last_block_logs_bloom: Bloom
    last_requests_hash: Bytes
    last_execution_reverted: bool


def apply_fork(old: BlockChain) -> BlockChain:
    """
    Transforms the state from the previous hard fork (`old`) into the block
    chain object for this hard fork and returns it.

    When forks need to implement an irregular state transition, this function
    is used to handle the irregularity. See the :ref:`DAO Fork <dao-fork>` for
    an example.

    Parameters
    ----------
    old :
        Previous block chain object.

    Returns
    -------
    new : `BlockChain`
        Upgraded block chain object for this hard fork.
    """
    return old


def get_last_256_block_hashes(chain: BlockChain) -> List[Hash32]:
    """
    Obtain the list of hashes of the previous 256 blocks in order of
    increasing block number.

    This function will return less hashes for the first 256 blocks.

    The ``BLOCKHASH`` opcode needs to access the latest hashes on the chain,
    therefore this function retrieves them.

    Parameters
    ----------
    chain :
        History and current state.

    Returns
    -------
    recent_block_hashes : `List[Hash32]`
        Hashes of the recent 256 blocks in order of increasing block number.
    """
    recent_blocks = chain.blocks[-255:]
    # TODO: This function has not been tested rigorously
    if len(recent_blocks) == 0:
        return []

    recent_block_hashes = []

    for block in recent_blocks:
        prev_block_hash = block.header.parent_hash
        recent_block_hashes.append(prev_block_hash)

    # We are computing the hash only for the most recent block and not for
    # the rest of the blocks as they have successors which have the hash of
    # the current block as parent hash.
    most_recent_block_hash = keccak256(rlp.encode(recent_blocks[-1].header))
    recent_block_hashes.append(most_recent_block_hash)

    return recent_block_hashes


def state_transition(chain: BlockChain, block: Block) -> None:
    """
    Attempts to apply a block to an existing block chain.

    All parts of the block's contents need to be verified before being added
    to the chain. Blocks are verified by ensuring that the contents of the
    block make logical sense with the contents of the parent block. The
    information in the block's header must also match the corresponding
    information in the block.

    To implement Ethereum, in theory clients are only required to store the
    most recent 255 blocks of the chain since as far as execution is
    concerned, only those blocks are accessed. Practically, however, clients
    should store more blocks to handle reorgs.

    Parameters
    ----------
    chain :
        History and current state.
    block :
        Block to apply to `chain`.
    """
    validate_block(chain, block)

    block_env = vm.BlockEnvironment(
        chain_id=chain.chain_id,
        state=chain.state,
        block_gas_limit=block.header.gas_limit,
        block_gas_used=block.header.gas_used,
        block_hashes=get_last_256_block_hashes(chain),
        coinbase=block.header.coinbase,
        number=block.header.number,
        base_fee_per_gas=block.header.base_fee_per_gas,
        time=block.header.timestamp,
        prev_randao=block.header.prev_randao,
        excess_blob_gas=block.header.excess_blob_gas,
        parent_beacon_block_root=block.header.parent_beacon_block_root,
    )

    block_output = apply_body(
        block_env=block_env,
        transactions=block.transactions,
        withdrawals=block.withdrawals,
    )

    transactions_root = root(block_output.transactions_trie)
    receipt_root = root(block_output.receipts_trie)
    block_logs_bloom = logs_bloom(block_output.block_logs)
    requests_hash = compute_requests_hash(block_output.requests)

    chain.last_block_logs_bloom = block_logs_bloom
    chain.last_transactions_root = transactions_root
    chain.last_receipt_root = receipt_root
    chain.last_requests_hash =  requests_hash
    chain.last_execution_reverted = block_output.execution_reverted
    chain.blocks.append(block)
    if len(chain.blocks) > 255:
        # Real clients have to store more blocks to deal with reorgs, but the
        # protocol only requires the last 255
        chain.blocks = chain.blocks[-255:]


def calculate_base_fee_per_gas(
    block_gas_limit: Uint,
    parent_gas_limit: Uint,
    parent_gas_used: Uint,
    parent_base_fee_per_gas: Uint,
) -> Uint:
    """
    Calculates the base fee per gas for the block.

    Parameters
    ----------
    block_gas_limit :
        Gas limit of the block for which the base fee is being calculated.
    parent_gas_limit :
        Gas limit of the parent block.
    parent_gas_used :
        Gas used in the parent block.
    parent_base_fee_per_gas :
        Base fee per gas of the parent block.

    Returns
    -------
    base_fee_per_gas : `Uint`
        Base fee per gas for the block.
    """
    parent_gas_target = parent_gas_limit // ELASTICITY_MULTIPLIER
    if not check_gas_limit(block_gas_limit, parent_gas_limit):
        raise InvalidBlock

    if parent_gas_used == parent_gas_target:
        expected_base_fee_per_gas = parent_base_fee_per_gas
    elif parent_gas_used > parent_gas_target:
        gas_used_delta = parent_gas_used - parent_gas_target

        parent_fee_gas_delta = parent_base_fee_per_gas * gas_used_delta
        target_fee_gas_delta = parent_fee_gas_delta // parent_gas_target

        base_fee_per_gas_delta = max(
            target_fee_gas_delta // BASE_FEE_MAX_CHANGE_DENOMINATOR,
            Uint(1),
        )

        expected_base_fee_per_gas = (
            parent_base_fee_per_gas + base_fee_per_gas_delta
        )
    else:
        gas_used_delta = parent_gas_target - parent_gas_used

        parent_fee_gas_delta = parent_base_fee_per_gas * gas_used_delta
        target_fee_gas_delta = parent_fee_gas_delta // parent_gas_target

        base_fee_per_gas_delta = (
            target_fee_gas_delta // BASE_FEE_MAX_CHANGE_DENOMINATOR
        )

        expected_base_fee_per_gas = (
            parent_base_fee_per_gas - base_fee_per_gas_delta
        )

    return Uint(expected_base_fee_per_gas)


def validate_header(chain: BlockChain, header: Header) -> None:
    """
    Verifies a block header.

    In order to consider a block's header valid, the logic for the
    quantities in the header should match the logic for the block itself.
    For example the header timestamp should be greater than the block's parent
    timestamp because the block was created *after* the parent block.
    Additionally, the block's number should be directly following the parent
    block's number since it is the next block in the sequence.
    The header should also have deferred execution outputs (pre_state_root,
    parent_receipt_root, parent_requests_hash etc...) matching the parent block.

    Parameters
    ----------
    chain :
        History and current state.
    header :
        Header to check for correctness.
    """
    if header.number < Uint(1):
        raise InvalidBlock

    parent_header = chain.blocks[-1].header

    excess_blob_gas = calculate_excess_blob_gas(parent_header)
    if header.excess_blob_gas != excess_blob_gas:
        raise InvalidBlock

    if header.gas_used > header.gas_limit:
        raise InvalidBlock
    if header.blob_gas_used > MAX_BLOB_GAS_PER_BLOCK:
        raise InvalidBlock

    # If parent block was reverted, use 0 as parent_gas_used
    parent_gas_used = Uint(0) if header.parent_execution_reverted else parent_header.gas_used
    expected_base_fee_per_gas = calculate_base_fee_per_gas(
        header.gas_limit,
        parent_header.gas_limit,
        parent_gas_used,
        parent_header.base_fee_per_gas,
    )
    if expected_base_fee_per_gas != header.base_fee_per_gas:
        raise InvalidBlock
    if header.timestamp <= parent_header.timestamp:
        raise InvalidBlock
    if header.number != parent_header.number + Uint(1):
        raise InvalidBlock
    if len(header.extra_data) > 32:
        raise InvalidBlock
    if header.difficulty != 0:
        raise InvalidBlock
    if header.nonce != b"\x00\x00\x00\x00\x00\x00\x00\x00":
        raise InvalidBlock
    if header.ommers_hash != EMPTY_OMMER_HASH:
        raise InvalidBlock

    block_parent_hash = keccak256(rlp.encode(parent_header))
    if header.parent_hash != block_parent_hash:
        raise InvalidBlock
    
    # validate deferred execution outputs from the parent
    if header.parent_transactions_root != chain.last_transactions_root:
        raise InvalidBlock
    if header.parent_receipt_root != chain.last_receipt_root:
        raise InvalidBlock
    if header.parent_bloom != chain.last_block_logs_bloom:
        raise InvalidBlock
    if header.parent_requests_hash != chain.last_requests_hash:
        raise InvalidBlock
    if header.pre_state_root != state_root(chain.state):
        raise InvalidBlock
    if header.parent_execution_reverted != chain.last_execution_reverted:
        raise InvalidBlock

def validate_block(chain: BlockChain, block: Block) -> None:
    """
    Verifies a block without executing its transactions.

    The function checks the header, validates every transaction, and 
    makes sure the transactions/withdrawals roots match the block data.
    Transaction validation includes checking that senders are 
    sufficiently funded for all their txs already in the 
    block's pre-state, and that nonces are sequential. The only
    field that cannot be verified upfront is `header.gas_used`.
    The block is valid as long as `validate_block` returns
    without raising an error, but, if the gas used is later found
    to be incorrect during execution, the entire execution reverts.
    

    Parameters
    ----------
    chain :
        History and current state.
    block :
        Block to validate.

    Raises
    ------
    InvalidBlock
        If the block fails any pre-execution verification.
    """
    total_inclusion_gas = Uint(0)
    total_blob_gas_used = Uint(0)
    withdrawals_trie: Trie[Bytes, Optional[Union[Bytes, Withdrawal]]] = Trie(
        secured=False, default=None
    )

    validate_header(chain, block.header)

    if block.ommers != ():
        raise InvalidBlock
    
    sender_balances = {}
    sender_nonces = {}
    blob_gas_price = calculate_blob_gas_price(block.header.excess_blob_gas)

    # First, collect all txs per sender, and their costs/nonces
    for i, tx in enumerate(map(decode_transaction, block.transactions)):
        validate_transaction(tx, block.header.base_fee_per_gas, block.header.excess_blob_gas)
        sender_address = recover_sender(chain.chain_id, tx)

        intrinsic_gas, calldata_floor_gas_cost = calculate_intrinsic_cost(tx)
        blob_gas_used = calculate_total_blob_gas(tx)

        total_inclusion_gas += max(intrinsic_gas, calldata_floor_gas_cost)
        total_blob_gas_used += blob_gas_used

        effective_gas_price = calculate_effective_gas_price(tx, block.header.base_fee_per_gas)
        max_gas_fee = tx.gas * effective_gas_price + blob_gas_used * blob_gas_price

        # Track total cost per sender
        if sender_address not in sender_balances:
            account = get_account(chain.state, sender_address)
            is_sender_eoa = (
                account.code == bytearray() 
                or is_valid_delegation(account.code)
            )
            if not is_sender_eoa:
                raise InvalidBlock
            sender_balances[sender_address] = account.balance
            sender_nonces[sender_address] = account.nonce
        if sender_balances[sender_address] < max_gas_fee + Uint(tx.value):
            raise InvalidBlock
        if sender_nonces[sender_address] != tx.nonce:
            raise InvalidBlock
        sender_balances[sender_address] -= max_gas_fee + Uint(tx.value)
        sender_nonces[sender_address] += 1

    if total_inclusion_gas > block.header.gas_used:
        raise InvalidBlock
    if total_blob_gas_used != block.header.blob_gas_used:
        raise InvalidBlock


    for i, wd in enumerate(block.withdrawals):
        trie_set(withdrawals_trie, rlp.encode(Uint(i)), rlp.encode(wd))

    if block.header.withdrawals_root != root(withdrawals_trie):
        raise InvalidBlock


def validate_transaction(
    tx: Transaction,
    base_fee_per_gas: U64,
    excess_blob_gas: U64,
) -> None:

    """
    Verifies a transaction.

    Checks include:
    * `tx.gas` ≥ minimum gas used by the tx (EIP-7623)
    * `tx.nonce` < 2**64-1 (EIP-2681)
    * `tx.data` ≤ 2 × MAX_CODE_SIZE (EIP-3860)
    * `tx.max_fee_per_gas` ≥ `tx.max_priority_fee_per_gas` (EIP-1559)
    * Fees sufficient to cover network base fees (EIP-1559 / 4844)
    * Blob tx: valid versioned hashes, and at least one (EIP-4844)

    Parameters
    ----------
    tx :
        Transaction to validate.
    base_fee_per_gas :
        Current block base fee.
    excess_blob_gas :
        Excess blob gas (for blob fee).

    Returns
    -------
    sender : ethereum.fork_types.Address
        Recovered sender address.

    Raises
    ------
    InvalidBlock
        If any rule is violated.
    """

    intrinsic_gas, calldata_floor_gas_cost = calculate_intrinsic_cost(tx)
    if max(intrinsic_gas, calldata_floor_gas_cost) > tx.gas:
        raise InvalidBlock
    if U256(tx.nonce) >= U256(U64.MAX_VALUE):
        raise InvalidBlock
    if tx.to == Bytes0(b"") and len(tx.data) > 2 * MAX_CODE_SIZE:
        raise InvalidBlock

    if isinstance(tx, (FeeMarketTransaction, BlobTransaction, SetCodeTransaction)):
        if tx.max_fee_per_gas < tx.max_priority_fee_per_gas:
            raise InvalidBlock
        if tx.max_fee_per_gas < base_fee_per_gas:
            raise InvalidBlock
    else:
        if tx.gas_price < base_fee_per_gas:
            raise InvalidBlock

    if isinstance(tx, BlobTransaction):
        if not isinstance(tx.to, Address):
            raise InvalidBlock
        if len(tx.blob_versioned_hashes) == 0:
            raise InvalidBlock
        for blob_versioned_hash in tx.blob_versioned_hashes:
            if blob_versioned_hash[0:1] != VERSIONED_HASH_VERSION_KZG:
                raise InvalidBlock
        blob_gas_price = calculate_blob_gas_price(excess_blob_gas)
        if Uint(tx.max_fee_per_blob_gas) < blob_gas_price:
            raise InvalidBlock
        
    if isinstance(tx, SetCodeTransaction):
        if not any(tx.authorizations):
            raise InvalidBlock


def make_receipt(
    tx: Transaction,
    error: Optional[EthereumException],
    cumulative_gas_used: Uint,
    logs: Tuple[Log, ...],
) -> Union[Bytes, Receipt]:
    """
    Make the receipt for a transaction that was executed.

    Parameters
    ----------
    tx :
        The executed transaction.
    error :
        Error in the top level frame of the transaction, if any.
    cumulative_gas_used :
        The total gas used so far in the block after the transaction was
        executed.
    logs :
        The logs produced by the transaction.

    Returns
    -------
    receipt :
        The receipt for the transaction.
    """
    receipt = Receipt(
        succeeded=error is None,
        cumulative_gas_used=cumulative_gas_used,
        bloom=logs_bloom(logs),
        logs=logs,
    )

    return encode_receipt(tx, receipt)


def process_system_transaction(
    block_env: vm.BlockEnvironment,
    target_address: Address,
    data: Bytes,
) -> MessageCallOutput:
    """
    Process a system transaction.

    Parameters
    ----------
    block_env :
        The block scoped environment.
    target_address :
        Address of the contract to call.
    data :
        Data to pass to the contract.

    Returns
    -------
    system_tx_output : `MessageCallOutput`
        Output of processing the system transaction.
    """
    system_contract_code = get_account(block_env.state, target_address).code

    tx_env = vm.TransactionEnvironment(
        origin=SYSTEM_ADDRESS,
        gas_price=block_env.base_fee_per_gas,
        gas=SYSTEM_TRANSACTION_GAS,
        access_list_addresses=set(),
        access_list_storage_keys=set(),
        transient_storage=TransientStorage(),
        blob_versioned_hashes=(),
        authorizations=(),
        index_in_block=None,
        tx_hash=None,
        traces=[],
    )

    system_tx_message = Message(
        block_env=block_env,
        tx_env=tx_env,
        caller=SYSTEM_ADDRESS,
        target=target_address,
        gas=SYSTEM_TRANSACTION_GAS,
        value=U256(0),
        data=data,
        code=system_contract_code,
        depth=Uint(0),
        current_target=target_address,
        code_address=target_address,
        should_transfer_value=False,
        is_static=False,
        accessed_addresses=set(),
        accessed_storage_keys=set(),
        disable_precompiles=False,
        parent_evm=None,
    )

    system_tx_output = process_message_call(system_tx_message)

    return system_tx_output


def apply_body(
    block_env: vm.BlockEnvironment,
    transactions: Tuple[Union[LegacyTransaction, Bytes], ...],
    withdrawals: Tuple[Withdrawal, ...],
) -> vm.BlockOutput:
    """
    Executes a block.

    Many of the contents of a block are stored in data structures called
    tries. There is a transactions trie which is similar to a ledger of the
    transactions stored in the current block. There is also a receipts trie
    which stores the results of executing a transaction, like the post state
    and gas used. This function creates and executes the block that is to be
    added to the chain.

    Parameters
    ----------
    block_env :
        The block scoped environment.
    transactions :
        Transactions included in the block.
    withdrawals :
        Withdrawals to be processed in the current block.

    Returns
    -------
    block_output :
        The block output for the current block.
    """
    block_output = vm.BlockOutput()

    process_system_transaction(
        block_env=block_env,
        target_address=BEACON_ROOTS_ADDRESS,
        data=block_env.parent_beacon_block_root,
    )

    process_system_transaction(
        block_env=block_env,
        target_address=HISTORY_STORAGE_ADDRESS,
        data=block_env.block_hashes[-1],  # The parent hash
    )

    process_transactions(block_env, block_output, transactions)

    process_withdrawals(block_env, block_output, withdrawals)

    process_general_purpose_requests(
        block_env=block_env,
        block_output=block_output,
    )

    return block_output

def process_transactions(block_env: vm.BlockEnvironment, block_output: vm.BlockOutput, transactions: Tuple[Union[LegacyTransaction, Bytes], ...]) -> None:
    begin_transaction(block_env.state)

    decoded_transactions = [decode_transaction(tx) for tx in transactions]

    # Charge senders the maximum fees upfront
    for tx in decoded_transactions:
        deduct_max_tx_fee_from_sender_balance(block_env, tx)

    for i, tx in enumerate(decoded_transactions):
        process_transaction(block_env, block_output, tx, Uint(i))
        if block_output.execution_reverted:
            break

    block_output.execution_reverted = (
        block_output.execution_reverted 
        or block_output.block_gas_used != block_env.block_gas_used
    )

    if block_output.execution_reverted:
        rollback_transaction(block_env.state)
        block_output.block_gas_used = Uint(0)
        block_output.transactions_trie = Trie(secured=False, default=None)
        block_output.receipts_trie = Trie(secured=False, default=None)
        block_output.receipt_keys = ()
        block_output.block_logs = ()
        block_output.requests = []
        block_output.execution_reverted = True
    else:
        commit_transaction(block_env.state)


def process_general_purpose_requests(
    block_env: vm.BlockEnvironment,
    block_output: vm.BlockOutput,
) -> None:
    """
    Process all the requests in the block.

    Parameters
    ----------
    block_env :
        The execution environment for the Block.
    block_output :
        The block output for the current block.
    """
    if block_output.execution_reverted:
        return
    
    # Requests are to be in ascending order of request type
    deposit_requests = parse_deposit_requests(block_output)
    requests_from_execution = block_output.requests
    if len(deposit_requests) > 0:
        requests_from_execution.append(DEPOSIT_REQUEST_TYPE + deposit_requests)

    system_withdrawal_tx_output = process_system_transaction(
        block_env=block_env,
        target_address=WITHDRAWAL_REQUEST_PREDEPLOY_ADDRESS,
        data=b"",
    )

    if len(system_withdrawal_tx_output.return_data) > 0:
        requests_from_execution.append(
            WITHDRAWAL_REQUEST_TYPE + system_withdrawal_tx_output.return_data
        )

    system_consolidation_tx_output = process_system_transaction(
        block_env=block_env,
        target_address=CONSOLIDATION_REQUEST_PREDEPLOY_ADDRESS,
        data=b"",
    )

    if len(system_consolidation_tx_output.return_data) > 0:
        requests_from_execution.append(
            CONSOLIDATION_REQUEST_TYPE
            + system_consolidation_tx_output.return_data
        )


def process_transaction(
    block_env: vm.BlockEnvironment,
    block_output: vm.BlockOutput,
    tx: Transaction,
    index: Uint,
) -> None:
    """
    Execute a transaction against the provided environment.

    This function processes the actions needed to execute a transaction.
    It decrements the sender's account after calculating the gas fee and
    refunds them the proper amount after execution. Calling contracts,
    deploying code, and incrementing nonces are all examples of actions that
    happen within this function or from a call made within this function.

    Accounts that are marked for deletion are processed and destroyed after
    execution.

    Parameters
    ----------
    block_env :
        Environment for the Ethereum Virtual Machine.
    block_output :
        The block output for the current block.
    tx :
        Transaction to execute.
    index:
        Index of the transaction in the block.
    """
    if block_output.block_gas_used + tx.gas > block_env.block_gas_limit:
        block_output.execution_reverted = True
        return
    
    trie_set(
        block_output.transactions_trie,
        rlp.encode(index),
        encode_transaction(tx),
    )

    intrinsic_gas, calldata_floor_gas_cost = calculate_intrinsic_cost(tx)
    sender = recover_sender(block_env.chain_id, tx)
    increment_nonce(block_env.state, sender)

    effective_gas_price = calculate_effective_gas_price(tx, block_env.base_fee_per_gas)
    blob_versioned_hashes = tx.blob_versioned_hashes if isinstance(tx, BlobTransaction) else ()
    gas = tx.gas - intrinsic_gas

    access_list_addresses = set()
    access_list_storage_keys = set()
    access_list_addresses.add(block_env.coinbase)
    if isinstance(
        tx,
        (
            AccessListTransaction,
            FeeMarketTransaction,
            BlobTransaction,
            SetCodeTransaction,
        ),
    ):
        for access in tx.access_list:
            access_list_addresses.add(access.account)
            for slot in access.slots:
                access_list_storage_keys.add((access.account, slot))

    authorizations: Tuple[Authorization, ...] = ()
    if isinstance(tx, SetCodeTransaction):
        authorizations = tx.authorizations

    tx_env = vm.TransactionEnvironment(
        origin=sender,
        gas_price=effective_gas_price,
        gas=gas,
        access_list_addresses=access_list_addresses,
        access_list_storage_keys=access_list_storage_keys,
        transient_storage=TransientStorage(),
        blob_versioned_hashes=blob_versioned_hashes,
        authorizations=authorizations,
        index_in_block=index,
        tx_hash=get_transaction_hash(encode_transaction(tx)),
        traces=[],
    )

    message = prepare_message(block_env, tx_env, tx)

    tx_output = process_message_call(message)

    # For EIP-7623 we first calculate the execution_gas_used, which includes
    # the execution gas refund.
    execution_gas_used = tx.gas - tx_output.gas_left
    gas_refund = min(
        execution_gas_used // Uint(5), Uint(tx_output.refund_counter)
    )
    execution_gas_used -= gas_refund

    # Transactions with less execution_gas_used than the floor pay at the
    # floor cost.
    tx_gas_used = max(execution_gas_used, calldata_floor_gas_cost)

    tx_output.gas_left = tx.gas - tx_gas_used
    gas_refund_amount = tx_output.gas_left * effective_gas_price

    # For non-1559 transactions effective_gas_price == tx.gas_price
    priority_fee_per_gas = effective_gas_price - block_env.base_fee_per_gas
    transaction_fee = tx_gas_used * priority_fee_per_gas

    # refund gas
    sender_balance_after_refund = get_account(
        block_env.state, sender
    ).balance + U256(gas_refund_amount)
    set_account_balance(block_env.state, sender, sender_balance_after_refund)

    # transfer miner fees
    coinbase_balance_after_mining_fee = get_account(
        block_env.state, block_env.coinbase
    ).balance + U256(transaction_fee)
    if coinbase_balance_after_mining_fee != 0:
        set_account_balance(
            block_env.state,
            block_env.coinbase,
            coinbase_balance_after_mining_fee,
        )
    elif account_exists_and_is_empty(block_env.state, block_env.coinbase):
        destroy_account(block_env.state, block_env.coinbase)

    for address in tx_output.accounts_to_delete:
        destroy_account(block_env.state, address)

    block_output.block_gas_used += tx_gas_used

    receipt = make_receipt(
        tx, tx_output.error, block_output.block_gas_used, tx_output.logs
    )

    receipt_key = rlp.encode(Uint(index))
    block_output.receipt_keys += (receipt_key,)

    trie_set(
        block_output.receipts_trie,
        receipt_key,
        receipt,
    )

    block_output.block_logs += tx_output.logs


def process_withdrawals(
    block_env: vm.BlockEnvironment,
    block_output: vm.BlockOutput,
    withdrawals: Tuple[Withdrawal, ...],
) -> None:
    """
    Increase the balance of the withdrawing account.
    """

    def increase_recipient_balance(recipient: Account) -> None:
        recipient.balance += wd.amount * U256(10**9)

    for i, wd in enumerate(withdrawals):
        modify_state(block_env.state, wd.address, increase_recipient_balance)

        if account_exists_and_is_empty(block_env.state, wd.address):
            destroy_account(block_env.state, wd.address)


def compute_header_hash(header: Header) -> Hash32:
    """
    Computes the hash of a block header.

    The header hash of a block is the canonical hash that is used to refer
    to a specific block and completely distinguishes a block from another.

    ``keccak256`` is a function that produces a 256 bit hash of any input.
    It also takes in any number of bytes as an input and produces a single
    hash for them. A hash is a completely unique output for a single input.
    So an input corresponds to one unique hash that can be used to identify
    the input exactly.

    Prior to using the ``keccak256`` hash function, the header must be
    encoded using the Recursive-Length Prefix. See :ref:`rlp`.
    RLP encoding the header converts it into a space-efficient format that
    allows for easy transfer of data between nodes. The purpose of RLP is to
    encode arbitrarily nested arrays of binary data, and RLP is the primary
    encoding method used to serialize objects in Ethereum's execution layer.
    The only purpose of RLP is to encode structure; encoding specific data
    types (e.g. strings, floats) is left up to higher-order protocols.

    Parameters
    ----------
    header :
        Header of interest.

    Returns
    -------
    hash : `ethereum.crypto.hash.Hash32`
        Hash of the header.
    """
    return keccak256(rlp.encode(header))


def check_gas_limit(gas_limit: Uint, parent_gas_limit: Uint) -> bool:
    """
    Validates the gas limit for a block.

    The bounds of the gas limit, ``max_adjustment_delta``, is set as the
    quotient of the parent block's gas limit and the
    ``GAS_LIMIT_ADJUSTMENT_FACTOR``. Therefore, if the gas limit that is
    passed through as a parameter is greater than or equal to the *sum* of
    the parent's gas and the adjustment delta then the limit for gas is too
    high and fails this function's check. Similarly, if the limit is less
    than or equal to the *difference* of the parent's gas and the adjustment
    delta *or* the predefined ``GAS_LIMIT_MINIMUM`` then this function's
    check fails because the gas limit doesn't allow for a sufficient or
    reasonable amount of gas to be used on a block.

    Parameters
    ----------
    gas_limit :
        Gas limit to validate.

    parent_gas_limit :
        Gas limit of the parent block.

    Returns
    -------
    check : `bool`
        True if gas limit constraints are satisfied, False otherwise.
    """
    max_adjustment_delta = parent_gas_limit // GAS_LIMIT_ADJUSTMENT_FACTOR
    if gas_limit >= parent_gas_limit + max_adjustment_delta:
        return False
    if gas_limit <= parent_gas_limit - max_adjustment_delta:
        return False
    if gas_limit < GAS_LIMIT_MINIMUM:
        return False

    return True


def calculate_effective_gas_price(tx: Transaction, base_fee_per_gas: Uint) -> Uint:
    """
    Calculates the effective gas price for a transaction.
    """
    if isinstance(tx, (FeeMarketTransaction, BlobTransaction, SetCodeTransaction)):
        priority_fee_per_gas = min(
            tx.max_priority_fee_per_gas,
            tx.max_fee_per_gas - base_fee_per_gas,
        )
        effective_gas_price = priority_fee_per_gas + base_fee_per_gas
    else:
        effective_gas_price = tx.gas_price
    return effective_gas_price


def deduct_max_tx_fee_from_sender_balance(block_env: vm.BlockEnvironment, tx: Transaction) -> None:
    effective_gas_price = calculate_effective_gas_price(tx, block_env.base_fee_per_gas)
    blob_gas_price = calculate_blob_gas_price(block_env.excess_blob_gas)
    blob_gas_used = calculate_total_blob_gas(tx)
    max_gas_fee = tx.gas * effective_gas_price + blob_gas_used * blob_gas_price
    sender = recover_sender(block_env.chain_id, tx)
    sender_account = get_account(block_env.state, sender)
    set_account_balance(block_env.state, sender, sender_account.balance - U256(max_gas_fee))