"""Freeze TTS speech inputs before baseline; never synthesize inside timed runs."""
import argparse
import base64
import hashlib
import io
import json
import wave
from pathlib import Path

from conversation_check import _http


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='https://127.0.0.1:19101')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cases={
        'direct': ['Hai cộng hai bằng mấy? Chỉ trả lời kết quả.',
                   'Thủ đô Việt Nam là gì? Trả lời thật ngắn.'],
        'clock':['Bây giờ là mấy giờ rồi?'],
        'search':['Thời tiết ở Hà Nội hôm nay thế nào? Hãy tra cứu giúp tôi.'],
        'long_context':['Hai cộng hai bằng mấy? Chỉ trả lời kết quả.'],
        'idle':['Thủ đô Việt Nam là gì? Trả lời thật ngắn.'],
        'natural':['Giải thích ngắn gọn tại sao bầu trời có màu xanh.'],
    }
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    manifest={}
    for name,texts in cases.items():
        rows=[]
        for i,text in enumerate(texts):
            reply=_http(args.base,'/try/tts',{'text':text,'voice':'quangminh'})
            wav=base64.b64decode(reply['wav_base64'])
            (output/f'{name}-{i}.wav').write_bytes(wav)
            with wave.open(io.BytesIO(wav)) as f:
                row={'reference':text,'sample_rate':f.getframerate(),
                     'pcm_base64':base64.b64encode(f.readframes(f.getnframes())).decode(),
                     'wav_sha256':hashlib.sha256(wav).hexdigest(),'search':name=='search'}
            if name == "direct":
                row["expected_answer_regex"] = r"(?:^|\D)4(?:$|\D)|bốn" if i == 0 else "hà nội"
            rows.append(row)
        manifest[name]=rows if name=='direct' else rows[0]
        print(name,flush=True)
    (output/'stimuli.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
