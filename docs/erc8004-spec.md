# ERC-8004 Standard Reference

ERC-8004 defines a standard for autonomous AI agent identity on Ethereum.

## Core Contracts

### IdentityRegistry (ERC-721 Extension)
- `mint(metadataURI)` — Register new agent, get sequential agentId
- `setAgentURI(agentId, uri)` — Update agent metadata
- `agentDID(agentId)` — Returns `eip155:{chainId}:{contract}:{id}`
- `agentIdOf(address)` — Lookup agent ID by address

### ReputationRegistry
- `giveFeedback(agentId, value, decimals, tag1, tag2, endpointURI, fileHash)` — Submit feedback
- `getFeedback(agentId, client)` — Get specific feedback
- `getAggregateScore(agentId)` — Returns (score, count)
- Prevents self-feedback (owner cannot rate own agent)

### ValidationRegistry
- `validationRequest(agentId, dataURI, dataHash)` — Submit request, returns requestId
- `submitValidation(requestId, response 0-100, commentURI)` — Validator responds
- `getValidationResult(requestId)` — Returns (validator, response, timestamp)
- `disputeValidation(requestId)` — Requester can dispute

### AgentWallet (EIP-1271)
- `isValidSignature(hash, signature)` — EIP-1271 verification
- `execute(target, value, data)` — Owner-only execution
- `executeWithSignature(...)` — EIP-712 meta-transaction
- Enables agent to sign on-chain actions

## Agent DID Format

```
eip155:{chainId}:{contractAddress}:{agentId}
```

Example:
```
eip155:11155111:0x1234abcd...:1
```

## Reputation Score Format

Scores use fixed-point with configurable decimals:
- `value=850, decimals=2` → score of 8.50
- `value=1000, decimals=3` → score of 1.000
- Aggregate score normalized to 2 decimal places

## Validation Flow

```
Agent executes trade
  → Creates trade artifact (JSON)
  → Computes keccak256 hash
  → Calls validationRequest(agentId, dataURI, hash)
  → Gets requestId
  → (Later) Calls submitValidation(requestId, score, "")
  → Score accumulates in ReputationRegistry
```

## References
- [ERC-8004 Proposal](https://eips.ethereum.org/EIPS/eip-8004)
- [lablab.ai Hackathon](https://lablab.ai)
