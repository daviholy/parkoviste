
class Common():
    args = None

    @classmethod
    def commonArguments(parser):
        parser.add_argument('--DEBUG', type=bool, default=False,
                        help='decides if script will run in debug mode (prints to stdout)')
        args = parser.parse_args()
        return args

    @classmethod
    def debug(self, name):
        def decorator(func):
            def inner(*arg):
                if self.args.DEBUG:
                    print(name)
                    func(*arg)
                    print()
                return
            return inner
        return decorator