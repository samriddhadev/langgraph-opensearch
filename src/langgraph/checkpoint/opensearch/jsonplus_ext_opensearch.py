import json
from typing import Any
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

class JSONPlusExtOSSerializer(JsonPlusSerializer):
    

    def dumps(self, obj: Any) -> str:
        return json.dumps(obj, default=self._default, ensure_ascii=False)

    def loads(self, data: str) -> Any:
        return json.loads(data, object_hook=self._reviver)