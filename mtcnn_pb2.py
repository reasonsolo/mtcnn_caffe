# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mtcnn.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='mtcnn.proto',
  package='',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x0bmtcnn.proto\"b\n\x05\x44\x61tum\x12\x0b\n\x03img\x18\x01 \x01(\x0c\x12\r\n\x05label\x18\x02 \x01(\x05\x12\x0c\n\x04\x62\x62ox\x18\x03 \x03(\x02\x12\x0e\n\x06landm5\x18\x04 \x03(\x02\x12\t\n\x01w\x18\x05 \x01(\x05\x12\t\n\x01h\x18\x06 \x01(\x05\x12\t\n\x01\x63\x18\x07 \x01(\x05')
)




_DATUM = _descriptor.Descriptor(
  name='Datum',
  full_name='Datum',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='img', full_name='Datum.img', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label', full_name='Datum.label', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bbox', full_name='Datum.bbox', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='landm5', full_name='Datum.landm5', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='w', full_name='Datum.w', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='h', full_name='Datum.h', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='c', full_name='Datum.c', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=15,
  serialized_end=113,
)

DESCRIPTOR.message_types_by_name['Datum'] = _DATUM
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Datum = _reflection.GeneratedProtocolMessageType('Datum', (_message.Message,), dict(
  DESCRIPTOR = _DATUM,
  __module__ = 'mtcnn_pb2'
  # @@protoc_insertion_point(class_scope:Datum)
  ))
_sym_db.RegisterMessage(Datum)


# @@protoc_insertion_point(module_scope)
