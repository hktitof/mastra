---
'@mastra/memory': patch
---

Knowledge curation no longer fails on Gemini models: the curator's `knowledge_update_node` tool schema was rejected by Google's API ("required only allowed for OBJECT type"), causing every curation attempt with a Gemini curator to fail before the model ran. The offending root-level union is gone from the tool schema. (The replacement shape landed separately — see the node-edit tool split in this same release.)
