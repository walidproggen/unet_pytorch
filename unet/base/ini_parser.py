import configparser
import ast


class Section:
    """
    Convenient class to store variables and their values.
    """
    def __init__(self):
        pass


class IniParser:
    """
    Class for reading config.ini files.
    Contents of INI-file will be accessible via dot convention.
    E.g. <section-name>.<variable-name>
    """

    def __init__(self, path):
        """
        Constructor.

        :param path: Path to the INI-file.
        """
        self.config = self._load_config(path)
        self.config_dict = self._create_config_dict()
        self._update_dict()

    def _load_config(self, path):
        """
        Loads an INI-file via configparser-lib.

        :param path: Path to the INI-file.
        :return: Returns ConfigParser-instance.
        """
        config = configparser.ConfigParser()
        config.read(path)
        return config

    def _create_config_dict(self):
        """
        Maps contents of loaded ConfigParser-instance to dict.

        :return: Dictionary with contents of the ini-file.
        """
        dicts = [dict(section) for section in dict(self.config).values()]
        return {section: item_dict for (section, item_dict) in zip(list(self.config), dicts)}

    def _str_to_type(self, str_input):
        """
        This func converts each string from the config to its most likely datatype by using ast.literal_eval.
        E.g. '2' -> int(2).

        :param str_input: input-string.
        :return: value with correct datatype extracted from input-string.
        """
        # Lists are being evaluated item by item
        if str_input[0] == '[':
            return [self._str_to_type(i.strip()) for i in str_input[1:-1].split(',')] if len(str_input) > 0 else []
        try:
            return ast.literal_eval(str_input)
        except:
            return str_input

    def _add_section(self, data):
        """
        Puts contents of <data> to the __dict__ of an Instance of Section-class.

        :param data: Dictionary.
        :return: Instance of Section-class, where contents of <data> can be accessed via dot-annotation.
        """
        data = {k: self._str_to_type(v) for k, v in data.items()}
        section = Section()
        vars(section).update(data)
        return section

    def _update_dict(self):
        """
        Puts contents of self.config_dict to the __dict__ of IniParser.

        :return: None
        """
        for key, value in self.config_dict.items():
            vars(self).update({key.lower(): self._add_section(value)})

