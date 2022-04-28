import json
from collections import namedtuple
from typing import List


class Stack:
    def __init__(self, id_: int, timestamp: int, clazz: List[str], frames: List[str],
                 message: List[str] = None, comment: str = None, issue_id: int = -1):
        self.id = id_
        self.ts = timestamp
        self.clazz = clazz
        self.frames = frames
        self.message = message or []
        self.comment = comment or ""

        self.issue_id = issue_id

    def eq_content(self, stack: 'Stack'):
        return self.clazz == stack.clazz and self.frames == stack.frames and \
               self.message == stack.message and self.comment == stack.comment

    @property
    def is_soe(self) -> bool:
        return self.clazz and max('StackOverflow' in cl for cl in self.clazz)

    @classmethod
    def from_json(cls, path: str, frames_field: str = 'frames', issue_id: int = -1) -> 'Stack':
        with open(path) as f:
            dict_ = json.loads(f.read())
            frames = dict_.get(frames_field, dict_["frames"])[0]
            return Stack(dict_['id'], dict_['timestamp'], dict_['class'], frames,
                         dict_.get('message', None), dict_.get('comment', None), issue_id=issue_id)

    def save_json(self, path: str):
        obj = {
            "id": self.id,
            "timestamp": self.ts,
            "class": self.clazz,
            "frames": [self.frames]
        }
        if self.message and self.message is not None:
            obj["message"] = self.message
        if self.comment and self.comment is not None:
            obj["comment"] = self.comment

        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)


StackEvent = namedtuple('StackEvent', 'id ts label')


class Issue:
    def __init__(self, id_: int, ts: int):
        self.id = id_
        self.stacks = {}
        self.last_update_ts = [ts]

    def add(self, st_id: int, ts: int, label: bool):
        if st_id in self.stacks:
            raise ValueError("stack already in this issue")
        self.stacks[st_id] = StackEvent(st_id, ts, label)
        self.update_ts(ts)

    def update_ts(self, ts: int):
        self.last_update_ts.append(ts)

    def remove(self, st_id: int, ts: int, label: bool):
        self.last_update_ts.remove(self.stacks[st_id].ts)
        del self.stacks[st_id]

    def confident_state(self) -> List[StackEvent]:
        return list(self.stacks.values())

    def last_ts(self) -> int:
        return self.last_update_ts[-1]
