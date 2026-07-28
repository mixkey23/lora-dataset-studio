> ### Two ways to install — and they do not update the same way
>
> **`git clone` is the one to prefer.** *Update & restart* then runs `git pull --ff-only`,
> so you get every fix the moment it lands — often days before it is packaged into a
> release like this one.
>
> **This ZIP is a snapshot of this tag.** *Update & restart* will only ever move you to
> the next release, so a fix shipped today reaches you whenever the next one is cut.
>
> ```
> git clone https://github.com/perfectgf/lora-dataset-studio.git
> cd lora-dataset-studio
> start.bat
> ```
>
> Already running a clone? You do not need the ZIP below.
