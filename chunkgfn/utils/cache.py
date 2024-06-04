class CachedPropertyWithInvalidation:
    """
    This is a decorator for cached properties with invalidation.
    It allows to cache a property value and invalidate the cache when a specific
    attribute changes. Keep in mind that the attribute change is captured by an equality
    check. If the attribute is mutable, the decorator will not be able to detect changes
    """

    def __init__(self, func, attr_name):
        self.func = func
        self.cache_name = f"_{func.__name__}_cache"
        self.attr_name = attr_name
        self.attr_cache_name = f"_{attr_name}_cache"

    def __get__(self, instance, owner):
        if instance is None:
            return self

        # Get the current value of the attribute
        current_attr_value = getattr(instance, self.attr_name)

        # Get the cached attribute value if it exists
        cached_attr_value = getattr(instance, self.attr_cache_name, None)

        if cached_attr_value != current_attr_value:
            # If the attribute values don't match, recalculate the property
            value = self.func(instance)
            setattr(instance, self.cache_name, value)
            setattr(instance, self.attr_cache_name, current_attr_value)
        elif not hasattr(instance, self.cache_name):
            # If the cache doesn't exist, calculate the property
            value = self.func(instance)
            setattr(instance, self.cache_name, value)
        else:
            # Otherwise, return the cached value
            value = getattr(instance, self.cache_name)

        return value


def cached_property_with_invalidation(attr_name):
    def decorator(func):
        return CachedPropertyWithInvalidation(func, attr_name)

    return decorator
