from marshmallow import fields, Schema, ValidationError


class BytesField(fields.Field):
    def _validate(self, value):
        if not isinstance(value, bytes):
            raise ValidationError('Invalid input type.')

        if value is None or value == b'':
            raise ValidationError('Invalid value')


class InferenceSchema(Schema):
    data = BytesField(attribute='data')
