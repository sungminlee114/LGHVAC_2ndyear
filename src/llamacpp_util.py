def wait_for_input_waiting(process, logger=None):
    lines = []
    before_char = ""
    line = ""
    while process.poll() is None:
        char = process.stdout.read(1)
        if not char:
            break
        elif f"{before_char}{char}" == "\n>":
            break
        
        if char == "\n":
            if logger is not None:
                logger.debug(line)
            lines.append(line)
            line = ""
        else:
            line += char

        before_char = char
    
    return "\n".join(lines).strip()
