class NoopContextManager:
    """Context manager that does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
