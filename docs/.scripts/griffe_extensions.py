"""
Griffe extension for the docs build that trims rendered signatures.

Parameter annotations are stripped so that definition lines stay compact,
e.g. `new(tensor, geometry=None) -> Volume`. Return annotations and default
values are kept (types are otherwise documented in the docstrings).

Methods wrapped by the custom caching decorators are converted into
read-only properties, matching their runtime behavior.
"""

import griffe


class TrimSignatures(griffe.Extension):

    def on_function_instance(self, *, func, **kwargs):
        # the custom caching decorators wrap methods into read-only properties,
        # so present them as attributes like griffe does for real properties
        decorators = {str(dec.value).split('.')[-1] for dec in func.decorators}
        if decorators & {'cached', 'cached_transferable'}:
            attribute = griffe.Attribute(
                name=func.name,
                lineno=func.lineno,
                endlineno=func.endlineno,
                annotation=func.returns,
                docstring=func.docstring,
                parent=func.parent,
            )
            attribute.labels.add('property')
            func.parent.set_member(func.name, attribute)
            return

        for parameter in func.parameters:
            parameter.annotation = None
