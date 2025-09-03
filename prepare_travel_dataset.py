"""
日本語旅行対話データセットの準備
GPT-5 nano用にフォーマット
"""

import json
from pathlib import Path
from typing import List, Dict

def create_travel_dataset():
    """
    日本語の旅行関連対話データを作成
    """
    
    # トレーニングデータ
    training_data = [
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "東京から京都への一番早い行き方を教えてください。"},
                {"role": "assistant", "content": "東京から京都への最速の移動方法は新幹線「のぞみ」です。東京駅から京都駅まで約2時間15分で到着します。料金は自由席13,320円、指定席14,170円です。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "北海道でおすすめの温泉地を教えてください。"},
                {"role": "assistant", "content": "北海道のおすすめ温泉地：1.登別温泉（地獄谷で有名）、2.洞爺湖温泉（湖畔の絶景）、3.定山渓温泉（札幌から近い）、4.層雲峡温泉（大雪山の自然）、5.阿寒湖温泉（アイヌ文化体験も可能）"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "沖縄旅行は何月がベストシーズンですか？"},
                {"role": "assistant", "content": "沖縄のベストシーズンは3月下旬〜4月と10月〜11月です。この時期は気候が安定し、台風の心配も少なく、観光に最適です。7-8月は海水浴には良いですが混雑し、5-6月は梅雨で雨が多めです。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "JRパスは外国人観光客だけが買えるのですか？"},
                {"role": "assistant", "content": "基本的には外国人観光客向けですが、海外在住の日本人も条件を満たせば購入可能です。購入には「短期滞在」の在留資格が必要で、日本国外の旅行会社で引換証を購入し、日本で交換します。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "富士山に登るのに必要な装備は？"},
                {"role": "assistant", "content": "富士山登山の必須装備：登山靴、雨具（上下分離型）、防寒着、ヘッドランプ、手袋、帽子、日焼け止め、水2L以上、行動食、小銭（トイレ用）、ゴミ袋。高山病対策も重要です。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "東京ディズニーランドのチケットは事前購入が必要ですか？"},
                {"role": "assistant", "content": "はい、現在は事前のオンライン購入が基本です。当日券は販売枚数に限りがあり、混雑日は売り切れることも。公式サイトやアプリから日付指定チケットの購入をおすすめします。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "成田空港から東京都心への移動方法を教えてください。"},
                {"role": "assistant", "content": "成田空港から都心への主な移動方法：1.成田エクスプレス（東京駅まで約60分、3,070円）、2.京成スカイライナー（上野まで約40分、2,570円）、3.リムジンバス（主要ホテル直通、約90分、3,100円）、4.京成本線（約70分、1,050円で最安）"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "日本の電車でICカードは全国共通で使えますか？"},
                {"role": "assistant", "content": "主要ICカード（Suica、PASMO、ICOCA等）は相互利用が可能で、全国の主要都市で使えます。ただし、一部地方路線では使えない場合があります。また、エリアをまたぐ利用（東京→大阪の在来線等）はできません。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "箱根の観光には何日必要ですか？"},
                {"role": "assistant", "content": "箱根観光は1泊2日が理想的です。1日目：箱根湯本→強羅→大涌谷→芦ノ湖（海賊船）、2日目：箱根神社→関所跡→美術館巡り。日帰りも可能ですが、温泉旅館でゆっくり過ごすのがおすすめです。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "日本でレンタカーを借りるには国際免許が必要ですか？"},
                {"role": "assistant", "content": "外国の方は国際運転免許証が必要です。ジュネーブ条約加盟国発行のものが有効。ドイツ、スイス、フランス、ベルギー、台湾等は別途翻訳文が必要。日本の免許をお持ちの方はそのまま利用可能です。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "京都の紅葉の見頃はいつですか？"},
                {"role": "assistant", "content": "京都の紅葉は11月中旬〜12月上旬が見頃です。名所により時期が異なり、高雄は11月上旬、嵐山・東山は11月中旬〜下旬、貴船・大原は11月上旬〜中旬がピークです。混雑を避けるなら平日の早朝がおすすめ。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "日本の新幹線で大きな荷物を持ち込めますか？"},
                {"role": "assistant", "content": "3辺合計160cm以内なら持ち込み可能です。それを超える特大荷物（250cm以内）は、事前予約制の「特大荷物スペースつき座席」が必要です。東海道・山陽・九州新幹線で適用。予約なしの場合1,000円の手数料がかかります。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "大阪で食べるべき名物料理は何ですか？"},
                {"role": "assistant", "content": "大阪の必食グルメ：たこ焼き（道頓堀）、お好み焼き（鶴橋）、串カツ（新世界）、きつねうどん（道頓堀）、551蓬莱の豚まん、りくろーおじさんのチーズケーキ。食い倒れの街を満喫してください！"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "USJのエクスプレスパスは必要ですか？"},
                {"role": "assistant", "content": "混雑日は強く推奨します。通常2-3時間待ちのアトラクションも、エクスプレスパスなら15-30分で乗れます。特に土日祝、長期休暇は必須。平日でも人気アトラクション優先なら購入価値があります。事前のオンライン購入がお得です。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "白川郷へのアクセス方法を教えてください。"},
                {"role": "assistant", "content": "白川郷への主なアクセス：1.高山から（濃飛バス50分、2,600円）、2.金沢から（濃飛・北鉄バス85分、2,000円）、3.名古屋から（岐阜バス3時間、4,000円）。冬季は予約必須。レンタカーは冬季は雪道注意。高山経由が便利です。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "日本で使えるポケットWiFiのレンタルはどこでできますか？"},
                {"role": "assistant", "content": "ポケットWiFiレンタル：1.空港カウンター（成田、羽田、関空等）で当日受取、2.事前オンライン予約で自宅・ホテル配送、3.家電量販店（ビックカメラ、ヨドバシ）。料金は1日500-1,500円。最近はeSIMも便利でお得です。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "広島の原爆ドームと宮島を1日で回れますか？"},
                {"role": "assistant", "content": "1日で両方回ることは可能です。朝一で原爆ドーム・平和記念資料館（2-3時間）→広島駅→JR＋フェリーで宮島へ（約1時間）→厳島神社・商店街（3-4時間）。ただし、ゆっくり見学したい場合は1泊2日がおすすめです。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "日光東照宮の拝観料はいくらですか？"},
                {"role": "assistant", "content": "日光東照宮の拝観料：大人1,300円、小中学生450円。これは東照宮単独の料金で、二社一寺共通券（東照宮・輪王寺・二荒山神社）なら大人2,400円でお得。宝物館は別途500円。音声ガイド500円もおすすめです。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "札幌雪まつりはいつ開催されますか？"},
                {"role": "assistant", "content": "札幌雪まつりは毎年2月上旬の約1週間開催されます（2024年は2月4日〜11日）。会場は大通公園、すすきの、つどーむの3か所。大通会場の大雪像は必見。防寒対策必須で、滑りにくい靴が重要。ホテルは早めの予約を。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "奈良公園の鹿にせんべいをあげる時の注意点は？"},
                {"role": "assistant", "content": "鹿せんべいの注意点：1.せんべいを見せると鹿が集まるので素早くあげる、2.じらすと服を引っ張られることも、3.あげ終わったら両手を広げて「もうない」とアピール、4.小さな子供は大人と一緒に、5.鹿せんべい以外の食べ物は絶対NG。お辞儀する鹿もいます！"}
            ]
        }
    ]
    
    # 検証用データ
    validation_data = [
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "鎌倉の観光スポットを教えてください。"},
                {"role": "assistant", "content": "鎌倉の主要観光スポット：鶴岡八幡宮、大仏（高徳院）、長谷寺、報国寺（竹の庭）、銭洗弁財天、小町通り。江ノ電で江の島まで足を延ばすのもおすすめ。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "青春18きっぷとは何ですか？"},
                {"role": "assistant", "content": "青春18きっぷはJR全線の普通・快速列車が1日乗り放題×5回分のお得な切符（12,050円）。年齢制限なし。春夏冬の期間限定販売。新幹線・特急は利用不可。時間に余裕がある旅におすすめ。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                {"role": "user", "content": "河口湖から富士山五合目へ行く方法は？"},
                {"role": "assistant", "content": "河口湖駅から富士スバルライン五合目行きバスが運行（約50分、片道1,570円、往復2,360円）。7-9月の登山シーズンは増便。マイカー規制期間はバス利用必須。高山病予防のため五合目で1時間程度の高度順応をおすすめ。"}
            ]
        }
    ]
    
    # データ保存
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # トレーニングデータを保存
    train_file = data_dir / "travel_train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for item in training_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    # 検証データを保存
    val_file = data_dir / "travel_validation.jsonl"
    with open(val_file, "w", encoding="utf-8") as f:
        for item in validation_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"✅ データセットを作成しました")
    print(f"  - トレーニング: {len(training_data)}件 → {train_file}")
    print(f"  - 検証: {len(validation_data)}件 → {val_file}")
    
    # サンプル表示
    print("\n📝 データサンプル:")
    with open(train_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        data = json.loads(first_line)
        print(json.dumps(data, ensure_ascii=False, indent=2)[:500] + "...")
    
    return train_file, val_file


if __name__ == "__main__":
    create_travel_dataset()