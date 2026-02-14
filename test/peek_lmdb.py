import lmdb
import pickle

def peek_lmdb(path):
    """查看 LMDB 数据库中的内容摘要。"""
    # 以只读模式打开数据库
    env = lmdb.open(path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        print(f"总条目数: {env.stat()['entries']}")
        # 遍历所有键值对
        for key, value in cursor:
            print(f"键 (Key): {key.decode()}")
            # 反序列化数据
            data = pickle.loads(value)
            # 打印元数据信息
            print(f"元数据 (Meta): {data.get('meta')}")
            # break  # 如果只想看第一个，可以取消注释此行

if __name__ == "__main__":
    # 指定要查看的 LMDB 路径
    peek_lmdb("test/test_lmdb")
