class SupportRole(object):
    LOCAL = "local"
    GUEST = "guest"
    HOST = "host"
    ARBITER = "arbiter"

    @classmethod
    def support_roles(cls):
        return [cls.LOCAL, cls.GUEST, cls.HOST, cls.ARBITER]


class LinkKey(object):
    DATA = "data"
    MODEL = "model"
    CACHE = "cache"

    @classmethod
    def get_all_link_keywords(cls):
        return [cls.DATA, cls.MODEL, cls.CACHE]
